import asyncio
import inspect
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import PurePosixPath
from typing import Any, Awaitable, Callable, Mapping, ParamSpec, Protocol, TypeAlias, TypeVar, cast, overload, runtime_checkable
from uuid import uuid4 as uuid

import modal

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')


SyncHandler: TypeAlias = Callable[[T], None]
AsyncHandler: TypeAlias = Callable[[T], Awaitable[None]]
Handler: TypeAlias = SyncHandler[T] | AsyncHandler[T]


@runtime_checkable
class AsyncCallable(Protocol[P, R]):
    """Represents an async callable specifically."""

    __call__: Callable[P, Awaitable[R]]
    __name__: str
    __module__: str
    __qualname__: str
    __annotations__: dict
    __doc__: str | None


log = logging.getLogger(__name__)

# A single Queue can contain [...] up to 5,000 items.
# https://modal.com/docs/reference/modal.Queue
MAX_LEN = 5_000


def coerce_to_async(fn: Callable[..., T | Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    if inspect.iscoroutinefunction(fn):
        return fn

    fn = cast(Callable[..., T], fn)

    @wraps(fn)
    async def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


@asynccontextmanager
async def run(app: modal.App, *, shutdown_timeout: float = 10, log_handler: Handler[str]):
    """
    Run a Modal app and display its stdout stream.

    This differs from `modal.enable_output`, in that this function only shows logs from inside the container.

    Args:
        app: The Modal app to run.
        shutdown_timeout: Number of seconds to wait for trailing logs after the app exits.
        log_handler: A function that processes logs. Will be called with each log line as it becomes available.
    """

    async def consume():
        async for output in app._logs.aio():
            if output == 'Stopping app - local entrypoint completed.\n':
                # Consume this infrastructure message
                continue
            log_handler(output)
            # No need to break: the loop should exit when the app is done

    # 1. Start the app
    # 2. Start consuming logs
    # 3. Yield control to the caller
    # 4. Wait for the logs to finish

    task = None
    try:
        async with app.run():
            task = asyncio.create_task(consume())
            yield

    finally:
        if task is not None:
            # Can't wait inside the context manager, because the app would still be running
            try:
                await asyncio.wait_for(task, timeout=shutdown_timeout)
            except asyncio.TimeoutError:
                log.warning(f"Logging task didn't complete within {shutdown_timeout}s timeout")


FnId: TypeAlias = tuple[str, str]
Partition: TypeAlias = str | None


@dataclass
class Call:
    """cloudpickle-friendly representation of a function call."""

    fn_id: FnId
    args: tuple
    kwargs: Mapping


class DispatchGroup:
    functions: dict[FnId, Callable[..., Awaitable]]

    def __init__(self):
        self.functions = {}

    def add(self, fn_id: FnId, fn: Callable):
        self.functions[fn_id] = coerce_to_async(fn)

    async def dispatch(self, call: Call) -> None:
        fn = self.functions[call.fn_id]
        await fn(*call.args, **call.kwargs)


class Experiment:
    inbound: modal.Queue
    functions: dict[Partition, DispatchGroup]
    stdout: Handler[str]
    volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]
    image: modal.Image | None

    def __init__(self, name: str):
        self.app = modal.App(name)
        queue_name = f'{name}-queue-{uuid().hex[:12]}'
        self.inbound = modal.Queue.from_name(queue_name, create_if_missing=True)
        self.functions = defaultdict(DispatchGroup)
        self.stdout = lambda s: print(s, end='')
        self.volumes = {}
        self.image = None

    @property
    def name(self):
        return self.app.name

    @asynccontextmanager
    async def run(self, shutdown_timeout: float = 10):
        self.inbound.hydrate()
        async with (
            run(self.app, shutdown_timeout=shutdown_timeout, log_handler=self.stdout),
            asyncio.TaskGroup() as tg,
        ):
            receivers = [
                _Receiver(self.functions[group].dispatch, self.inbound, group)
                for group in self.functions]  # fmt: skip
            tasks = [
                tg.create_task(receiver.consume())
                for receiver in receivers]  # fmt: skip

            try:
                yield
            finally:
                for receiver in receivers:
                    receiver.stop()
                if tasks:
                    _, pending = await asyncio.wait(tasks, timeout=shutdown_timeout)
                    if pending:
                        log.warning(f"{len(pending)} consumers didn't complete within {shutdown_timeout}s timeout")
                        for task in pending:
                            task.cancel()

    @overload
    def hither(self, func: Callable[P, Any], /) -> Callable[P, None]: ...
    @overload
    def hither(self, *, group: str | None = None) -> Callable[[Callable[P, Any]], Callable[P, None]]: ...

    def hither(
        self, func=None, *, group: str | None = None
    ) -> Callable[P, None] | Callable[[Callable[P, Any]], Callable[P, None]]:
        def decorator(fn: Callable[P, Any]) -> Callable[P, None]:
            fn_id = str(fn.__name__), uuid().hex[:12]
            send_from_remote = wraps(fn)(_sender(key=fn_id, queue=self.inbound, partition=group))
            self.functions[group].add(fn_id, fn)
            return send_from_remote

        if func is not None:
            return decorator(func)

        return decorator

    @overload
    def thither(self, func: AsyncCallable[P, R], /) -> AsyncCallable[P, R]: ...
    @overload
    def thither(self, **kwargs) -> Callable[[AsyncCallable[P, R]], AsyncCallable[P, R]]: ...

    def thither(
        self, func=None, **kwargs
    ) -> AsyncCallable[P, R] | Callable[[AsyncCallable[P, R]], AsyncCallable[P, R]]:
        def decorator(fn: AsyncCallable[P, R]) -> AsyncCallable[P, R]:
            if 'image' not in kwargs:
                kwargs['image'] = self.image

            volumes = kwargs.get('volumes', None) or {}
            kwargs['volumes'] = {**self.volumes, **volumes}

            @self.app.function(**kwargs)
            @wraps(fn)
            async def remote(*args, **kwargs):
                return await fn(*args, **kwargs)

            return remote.remote.aio

        if func is not None:
            return decorator(func)

        return decorator


def _sender[P](*, key: FnId, queue: modal.Queue, partition: Partition):
    # Define the worker send function here, to avoid having the real function in the closure.
    # Otherwise, all of its own closure variables would be pickled and sent to the worker.
    signal_partition = _signal_partition(partition)

    def send(*args, **kwargs):
        call = Call(key, args, kwargs)
        queue.put(call, partition=partition)
        queue.put(True, partition=signal_partition)

    return send


class _Receiver:
    def __init__(self, receive: AsyncHandler[T], queue: modal.Queue, partition: Partition):
        self._stop_event = asyncio.Event()
        self.queue = queue
        self.partition = partition
        self.receive = receive
        self._signal_partition = _signal_partition(partition)

    async def consume(self) -> None:
        """Take values from the queue until the context manager exits."""
        # This function is not exposed, so there's exactly one consumer.
        # It always runs locally.

        while True:
            # Wait until values are produced or the context manager exits.
            get_task = asyncio.create_task(self.queue.get_many.aio(MAX_LEN, partition=self._signal_partition))
            stop_task = asyncio.create_task(self._stop_event.wait())
            done, _ = await asyncio.wait(
                [get_task, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Either way, process all available messages.
            values = await self.queue.get_many.aio(MAX_LEN, block=False, partition=self.partition)
            for value in values:
                await self.receive(value)

            # Can't just check stop_event.is_set here; we need to know
            # whether it was set before the last batch.
            if stop_task in done:
                await self.queue.clear.aio(all=True)
                break

    def stop(self):
        """Stop the consumer."""
        self._stop_event.set()


def _signal_partition(partition: Partition) -> str:
    if partition is None:
        return '__signal'
    if partition.endswith('__signal'):
        raise ValueError('Partition name cannot end with "__signal"')
    return partition + '__signal'

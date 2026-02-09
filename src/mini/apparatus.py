"""
Executor protocol and progress reporting infrastructure.

Provides shared components for executor implementations:
- ``Executor`` protocol defining the map interface
- ``emit_progress()`` for jobs to report progress

Example::

    from mini.progress import emit_progress

    def train(config):
        for epoch in range(100):
            ...
            emit_progress(epoch, 100, message=f"loss={loss:.4f}")
        return result
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Iterable, Iterator, ParamSpec, TypeVar

P = ParamSpec('P')
R = TypeVar('R')


# ---------------------------------------------------------------------------
# Apparatus protocol
# ---------------------------------------------------------------------------


class Apparatus(ABC):
    """Protocol for running a function over a sweep of inputs."""

    def run(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Run a single function and return its result."""

        @wraps(fn)
        def wrapper(_) -> R:
            return fn(*args, **kwargs)

        return next(self.map(wrapper, [None]))

    @abstractmethod
    def amap(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[R, None]:
        """
        Map *fn* over one or more iterables.

        Like ``concurrent.futures.Executor.map`` and Modal's ``Function.map``:
        the iterables are zipped together and each tuple is unpacked as
        positional arguments.  *kwargs* (if given) are forwarded to every
        call.

        ::

            executor.map(fn, [1, 2, 3])                    # fn(1), fn(2), fn(3)
            executor.map(fn, [1, 2], ['a', 'b'])            # fn(1, 'a'), fn(2, 'b')
            executor.map(fn, [1, 2], kwargs={'k': 'v'})     # fn(1, k='v'), fn(2, k='v')
        """
        ...

    def map(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> Iterator[R]:
        """
        Map *fn* over one or more iterables.

        Like ``concurrent.futures.Executor.map`` and Modal's ``Function.map``:
        the iterables are zipped together and each tuple is unpacked as
        positional arguments.  *kwargs* (if given) are forwarded to every
        call.

        ::

            executor.map(fn, [1, 2, 3])                    # fn(1), fn(2), fn(3)
            executor.map(fn, [1, 2], ['a', 'b'])            # fn(1, 'a'), fn(2, 'b')
            executor.map(fn, [1, 2], kwargs={'k': 'v'})     # fn(1, k='v'), fn(2, k='v')
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            yield from _map_with_new_loop(self, fn, *iterables, kwargs=kwargs)
            return

        yield from _map_in_thread(self, fn, *iterables, kwargs=kwargs)

    @abstractmethod
    def before_each(self, hook: Callable[[], None]) -> Apparatus:
        """
        Return a new executor that runs *hook* before each job.

        This is useful for things like configuring logging or setting random
        seeds on a per-job basis.
        """
        ...


def _map_in_thread(
    executor: Apparatus,
    fn: Callable[..., R],
    *iterables: Iterable[Any],
    kwargs: dict[str, Any] | None,
) -> Iterator[R]:
    import threading
    import queue as queue_module

    results_queue: queue_module.Queue = queue_module.Queue()
    exception_holder: list[Exception] = []

    def run_in_thread():
        try:

            async def collect():
                async for result in executor.amap(fn, *iterables, kwargs=kwargs):
                    results_queue.put(('result', result))
                results_queue.put(('done', None))

            asyncio.run(collect())
        except Exception as e:
            exception_holder.append(e)
            results_queue.put(('error', e))

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    while True:
        msg_type, value = results_queue.get()
        if msg_type == 'result':
            yield value
        elif msg_type == 'done':
            break
        elif msg_type == 'error':
            raise value

    thread.join(timeout=1.0)
    if exception_holder:
        raise exception_holder[0]


def _map_with_new_loop(
    executor: Apparatus,
    fn: Callable[..., R],
    *iterables: Iterable[Any],
    kwargs: dict[str, Any] | None,
) -> Iterator[R]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        gen = executor.amap(fn, *iterables, kwargs=kwargs)
        while True:
            try:
                yield loop.run_until_complete(gen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()

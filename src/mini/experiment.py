import asyncio
import logging
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import PurePosixPath
from typing import Callable, ParamSpec, TypeVar, overload

import modal

from mini.types import AsyncCallable, Handler

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')


log = logging.getLogger(__name__)


@asynccontextmanager
async def run_with_logs(app: modal.App, *, shutdown_timeout: float = 10, log_handler: Handler[str]):
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


class Experiment:
    stdout: Handler[str]
    volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]
    image: modal.Image | None

    def __init__(self, name: str):
        self.app = modal.App(name)
        self.stdout = lambda s: print(s, end='')
        self.volumes = {}
        self.image = None

    @property
    def name(self):
        return self.app.name

    @asynccontextmanager
    async def __call__(self):
        async with run_with_logs(self.app, log_handler=self.stdout) as log_context:
            yield log_context

    @overload
    def run_thither(self, func: AsyncCallable[P, R], /) -> AsyncCallable[P, R]: ...
    @overload
    def run_thither(self, **kwargs) -> Callable[[AsyncCallable[P, R]], AsyncCallable[P, R]]: ...

    def run_thither(
        self, func=None, **kwargs
    ) -> AsyncCallable[P, R] | Callable[[AsyncCallable[P, R]], AsyncCallable[P, R]]:
        """
        Decorate a function to always run remotely.

        Args:
            func: The function to decorate.
            **kwargs: Arguments to pass to `modal.App.function`.

        Returns:
            remote: A function that executes on a remote worker. Must be called
            inside a `run` context manager.
        """

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

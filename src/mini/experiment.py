import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import PurePosixPath
from typing import Callable, Literal, ParamSpec, TypeVar, overload

import modal

from mini.guards import after, before
from mini.hither import run_hither
from mini.types import (
    AfterGuardDecorator,
    AsyncCallable,
    BeforeGuardDecorator,
    Guard,
    GuardContext,
    GuardContextFn,
    GuardDecorator,
    GuardExc,
    GuardFn,
    GuardFnExc,
    Handler,
)

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
    """A distributed experiment runner."""

    stdout: Handler[str]
    volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]
    image: modal.Image | None

    # Guards are context managers that are run remotely with each function.
    # Because they run for all functions, the function type parameter is not known at this point.
    # But GuardContextFn can also be used to create a guard for a specific function.
    guards: list[GuardContext | GuardContextFn[...]]

    def __init__(self, name: str):
        self.app = modal.App(name)
        self.stdout = lambda s: print(s, end='')
        self.volumes = {}
        self.image = None
        self.guards = []
        self.hither = run_hither

    @property
    def name(self):
        return self.app.name

    @asynccontextmanager
    async def __call__(self):
        async with run_with_logs(self.app, log_handler=self.stdout) as log_context:
            yield log_context

    @overload
    def guard(self, guard: GuardContext, /) -> GuardContext: ...

    @overload
    def guard(self, guard: GuardContextFn[P], /) -> GuardContextFn[P]: ...

    @overload
    def guard(self, *, placement: Literal['wrap'] = 'wrap') -> GuardDecorator[P]: ...

    @overload
    def guard(self, *, placement: Literal['before']) -> BeforeGuardDecorator[P]: ...

    @overload
    def guard(self, *, placement: Literal['after']) -> AfterGuardDecorator[P]: ...

    def guard(
        self,
        guard: GuardContext | GuardContextFn[P] | None = None,
        *,
        placement: Literal['before', 'after', 'wrap'] = 'wrap',
    ) -> GuardContext | GuardContextFn[P] | GuardDecorator[P] | BeforeGuardDecorator[P] | AfterGuardDecorator[P]:
        """
        Register a guard to run around the function.

        Guards are context managers that are run remotely with each function.
        They are entered before the function is called and exited after it
        returns. They can be used to set up and tear down resources.
        """
        if guard is not None:
            self.guards.append(guard)
            return guard

        if placement == 'wrap':
            return self.guard
        elif placement == 'before':
            return self._guard_before
        elif placement == 'after':
            return self._guard_after
        else:
            raise ValueError(f'Invalid placement: {placement}')

    @overload
    def _guard_before(self, callback: Guard) -> GuardContext: ...
    @overload
    def _guard_before(self, callback: GuardFn[P]) -> GuardContextFn[P]: ...

    def _guard_before(self, callback: Guard | GuardFn[P]) -> GuardContext | GuardContextFn[P]:
        guard_fn = before(callback)
        self.guards.append(guard_fn)
        return guard_fn

    @overload
    def _guard_after(self, callback: Guard) -> GuardContext: ...
    @overload
    def _guard_after(self, callback: GuardFn[P]) -> GuardContextFn[P]: ...
    @overload
    def _guard_after(self, callback: GuardExc) -> GuardContext: ...
    @overload
    def _guard_after(self, callback: GuardFnExc[P]) -> GuardContextFn[P]: ...

    def _guard_after(self, callback: Guard | GuardFn[P] | GuardExc | GuardFnExc[P]) -> GuardContext | GuardContextFn[P]:
        guard_fn = after(callback)
        self.guards.append(guard_fn)
        return guard_fn

    def before_each(self, callback: Guard | GuardFn[P]) -> Guard | GuardFn[P]:
        """Register a callback to run in the remote container before each function."""
        # Like _guard_before, but returns the unmodified callback so the type isn't changed
        self._guard_before(callback)
        return callback

    def after_each(
        self, callback: Guard | GuardFn[P] | GuardExc | GuardFnExc[P]
    ) -> Guard | GuardFn[P] | GuardExc | GuardFnExc[P]:
        """Register a callback to run in the remote container after each function."""
        # Like _guard_after, but returns the unmodified callback so the type isn't changed
        self._guard_after(callback)
        return callback

    @overload
    def thither(self, func: AsyncCallable[P, R], /) -> AsyncCallable[P, R]: ...
    @overload
    def thither(
        self,
        *,
        guards: list[GuardContext | GuardContextFn[P]] | None = None,
        **kwargs,
    ) -> Callable[[AsyncCallable[P, R]], AsyncCallable[P, R]]: ...

    def thither(
        self,
        func: AsyncCallable[P, R] | None = None,
        *,
        guards: list[GuardContext | GuardContextFn[P]] | None = None,
        **kwargs,
    ) -> AsyncCallable[P, R] | Callable[[AsyncCallable[P, R]], AsyncCallable[P, R]]:
        """
        Decorate a function to always run remotely.

        Args:
            func: The function to decorate.
            guards: Guards to run around the function. These are context managers that
                will be entered before the function is called and exited after it returns.
                They can be used to set up and tear down resources.
            **kwargs: Arguments to pass to `modal.App.function`.

        Returns:
            remote: A function that executes on a remote worker. Must be called
            inside a `run` context manager.
        """
        global_guards = self.guards
        specific_guards = guards or []

        def decorator(fn: AsyncCallable[P, R]) -> AsyncCallable[P, R]:
            if 'image' not in kwargs:
                kwargs['image'] = self.image

            volumes = kwargs.get('volumes', None) or {}
            kwargs['volumes'] = {**self.volumes, **volumes}

            @self.app.function(**kwargs)
            @wraps(fn)
            async def remote(*args, **kwargs):
                async def guarded_fn() -> R:
                    return await fn(*args, **kwargs)

                for guard in reversed(specific_guards):
                    guarded_fn = _wrap_with_guard(guarded_fn, guard, fn)

                for guard in reversed(global_guards):
                    guarded_fn = _wrap_with_guard(guarded_fn, guard, fn)

                return await guarded_fn()

            return remote.remote.aio

        if func is not None:
            return decorator(func)

        return decorator


def _wrap_with_guard(
    async_fn: AsyncCallable[[], R],
    guard: GuardContext | GuardContextFn[P] | GuardContextFn[...],
    original_fn: AsyncCallable[P, R],
):
    """Wrap an async function with a guard context manager."""

    @wraps(async_fn)
    async def wrapped(*args, **kwargs) -> R:
        if len(inspect.signature(guard).parameters) == 0:
            ctor_args = ()
        else:
            ctor_args = (original_fn,)

        with guard(*ctor_args):
            return await async_fn(*args, **kwargs)

    return wrapped

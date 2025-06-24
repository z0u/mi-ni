import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import PurePosixPath
from typing import AsyncGenerator, Callable, Literal, ParamSpec, TypeVar, Union, overload, override

import modal

from mini._modal.metadata import get_metadata
from mini._modal.model import FD, AppInfo, LogsItem, StateUpdate
from mini._modal.output import basic_output_handler, stream_logs
from mini._modal.runner import RunEventHandler, run_app
from mini._modal.task_state import TaskStateTracker, app_state_vis
from mini.guards import after, before
from mini.hither import run_hither
from mini.types import (
    AfterGuardDecorator,
    AfterGuardDecoratorFn,
    AsyncCallable,
    BeforeGuardDecorator,
    BeforeGuardDecoratorFn,
    Guard,
    GuardContext,
    GuardContextFn,
    GuardDecorator,
    GuardExc,
    GuardFn,
    GuardFnExc,
)

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')


log = logging.getLogger(__name__)

# patch_app()


class Experiment:
    """A distributed experiment runner."""

    output_handler: Callable[[LogsItem], None]
    volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]
    """Default volumes to use for all @thither functions."""
    image: modal.Image | None
    """Default image to use for all @thither functions."""

    # Guards are context managers that are run remotely with each function.
    # Because they run for all functions, the function type parameter is not known at this point.
    # But GuardContextFn can also be used to create a guard for a specific function.
    guards: list[GuardContext | GuardContextFn[...]]

    def __init__(self, name: str):
        if not name:
            raise ValueError('Experiment name must not be empty')
        self.app = modal.app.App(name)
        self.output_handler = basic_output_handler
        self.volumes = {}
        self.image = None
        self.guards = []
        self.hither = run_hither

    @property
    def name(self):
        return self.app.name

    @asynccontextmanager
    async def __call__(self, shutdown_timeout: float = 10):  # noqa: C901
        async def handle_progress(app_info: AppInfo):
            update_state_vis = app_state_vis(app_info)
            stateTracker = TaskStateTracker()
            async for item in stream_logs(self.app):
                match item:
                    case LogsItem(fd=FD.INFO):
                        await update_state_vis(message=item.data.strip())
                    case LogsItem():
                        self.output_handler(item)
                    case StateUpdate():
                        stateTracker.update(item)
                        await update_state_vis(stateTracker.tasks)

        # 1. Start the app
        # 2. Start consuming logs
        # 3. Yield control to the caller
        # 4. Wait for the logs to finish

        class Handler(RunEventHandler):
            @override
            async def on_init(self, app, ctx) -> None:
                pass

            @override
            async def on_create(self, app, ctx) -> None:
                pass

            @override
            @asynccontextmanager
            async def on_interrupt(self) -> AsyncGenerator[None, None]:
                yield

        task = None
        try:
            async with run_app(self.app, handler=Handler()):
                app_info = get_metadata(self.app)
                task = asyncio.create_task(handle_progress(app_info))
                yield

        except modal.exception.AuthError as e:
            e.add_note('Tip: Run `./go auth` to authenticate with Modal. If this is a notebook, restart the kernel.')
            raise

        finally:
            if task is not None:
                # Can't wait inside the context manager, because the app would still be running
                try:
                    await asyncio.wait_for(task, timeout=shutdown_timeout)
                except asyncio.TimeoutError:
                    log.warning(f"Output streaming task didn't complete within {shutdown_timeout}s timeout")

    @overload
    def thither(self, func: AsyncCallable[P, R], /) -> AsyncCallable[P, R]: ...
    @overload
    def thither(
        self,
        *,
        guards: list[GuardContext | GuardContextFn[P]] | None = None,
        **kwargs,
    ) -> Callable[[AsyncCallable[P, R]], AsyncCallable[P, R]]: ...

    def thither(  # noqa: C901
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
            async def modal_function(*_args, **_kwargs):
                # This is called in the remote container

                async def guarded_fn() -> R:
                    return await fn(*_args, **_kwargs)

                for guard in reversed(specific_guards):
                    guarded_fn = _wrap_with_guard(guarded_fn, guard, fn)

                for guard in reversed(global_guards):
                    guarded_fn = _wrap_with_guard(guarded_fn, guard, fn)

                return await guarded_fn()

            @wraps(fn)
            async def local_wrapper(*args, **kwargs):
                # This is called locally to start the remote function
                return await modal_function.remote.aio(*args, **kwargs)

            return local_wrapper

        if func is not None:
            return decorator(func)

        return decorator

    @overload
    def guard(self, guard: GuardContext, /) -> GuardContext: ...
    @overload
    def guard(self, guard: GuardContextFn[P], /) -> GuardContextFn[P]: ...
    @overload
    def guard(self, *, placement: Literal['wrap'] = 'wrap') -> GuardDecorator[P]: ...
    @overload
    def guard(self, *, placement: Literal['before']) -> BeforeGuardDecorator[P] | BeforeGuardDecoratorFn[P, R]: ...
    @overload
    def guard(self, *, placement: Literal['after']) -> AfterGuardDecorator[P] | AfterGuardDecoratorFn[P, R]: ...

    def guard(
        self,
        guard: GuardContext | GuardContextFn[P] | None = None,
        *,
        placement: Literal['before', 'after', 'wrap'] = 'wrap',
    ) -> Union[
        GuardContext,
        GuardContextFn[P],
        GuardDecorator[P],
        BeforeGuardDecorator[P],
        BeforeGuardDecoratorFn[P, R],
        AfterGuardDecorator[P],
        AfterGuardDecoratorFn[P, R],
    ]:
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
    def _guard_before(self, callback: Guard[R]) -> GuardContext: ...
    @overload
    def _guard_before(self, callback: GuardFn[P, R]) -> GuardContextFn[P]: ...

    def _guard_before(self, callback: Guard[R] | GuardFn[P, R]) -> GuardContext | GuardContextFn[P]:
        guard_fn = before(callback)
        self.guards.append(guard_fn)
        return guard_fn

    @overload
    def _guard_after(self, callback: Guard[R]) -> GuardContext: ...
    @overload
    def _guard_after(self, callback: GuardFn[P, R]) -> GuardContextFn[P]: ...
    @overload
    def _guard_after(self, callback: GuardExc[R]) -> GuardContext: ...
    @overload
    def _guard_after(self, callback: GuardFnExc[P, R]) -> GuardContextFn[P]: ...

    def _guard_after(
        self, callback: Guard[R] | GuardFn[P, R] | GuardExc[R] | GuardFnExc[P, R]
    ) -> GuardContext | GuardContextFn[P]:
        guard_fn = after(callback)
        self.guards.append(guard_fn)
        return guard_fn

    @overload
    def before_each(self, callback: Guard[R]) -> Guard[R]: ...
    @overload
    def before_each(self, callback: GuardFn[P, R]) -> GuardFn[P, R]: ...

    def before_each(self, callback: Guard[R] | GuardFn[P, R]) -> Guard[R] | GuardFn[P, R]:
        """Register a callback to run in the remote container before each function."""
        # Like _guard_before, but returns the unmodified callback so the type isn't changed
        self._guard_before(callback)
        return callback

    @overload
    def after_each(self, callback: Guard[R]) -> Guard[R]: ...
    @overload
    def after_each(self, callback: GuardFn[P, R]) -> GuardFn[P, R]: ...
    @overload
    def after_each(self, callback: GuardExc[R]) -> GuardExc[R]: ...
    @overload
    def after_each(self, callback: GuardFnExc[P, R]) -> GuardFnExc[P, R]: ...

    def after_each(
        self, callback: Guard[R] | GuardFn[P, R] | GuardExc[R] | GuardFnExc[P, R]
    ) -> Guard[R] | GuardFn[P, R] | GuardExc[R] | GuardFnExc[P, R]:
        """Register a callback to run in the remote container after each function."""
        # Like _guard_after, but returns the unmodified callback so the type isn't changed
        self._guard_after(callback)
        return callback


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

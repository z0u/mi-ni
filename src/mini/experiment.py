import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import PurePosixPath
from typing import Callable, Literal, ParamSpec, TypeVar, Union, overload
from uuid import uuid4 as uuid

import modal

from mini._state import CallState, CallStateError, CallTracker
from mini.guards import after, before
from mini.hither import Callback, run_hither
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
from mini.urns import is_mini_urn, short_id

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')


log = logging.getLogger(__name__)


class Experiment:
    """A distributed experiment runner."""

    output_handler: Callback[str]
    volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]
    """Default volumes to use for all @thither functions."""
    image: modal.Image | None
    """Default image to use for all @thither functions."""

    # Guards are context managers that are run remotely with each function.
    # Because they run for all functions, the function type parameter is not known at this point.
    # But GuardContextFn can also be used to create a guard for a specific function.
    guards: list[GuardContext | GuardContextFn[...]]

    def __init__(self, name: str):
        self.app = modal.App(name)
        self.output_handler = lambda s: print(s, end='')
        self.volumes = {}
        self.image = None
        self.guards = []
        self.hither = run_hither
        self._run_id: str | None = None

    @property
    def name(self):
        return self.app.name

    @asynccontextmanager
    async def __call__(self, shutdown_timeout: float = 10):  # noqa: C901
        async def consume(run_id: str):
            fn_tracker = CallTracker(run_id)

            async for output in self.app._logs.aio():
                lines = output.splitlines(keepends=True)
                for line in lines:
                    if is_mini_urn(line):
                        if CallState.matches(line.strip()):
                            try:
                                fn_tracker.handle(CallState.from_urn(line.strip()))
                            except CallStateError as e:
                                log.error('Call state error: %s', e)
                            continue

                    if fn_tracker.any_running():
                        # Only print output if there are running functions to avoid printing infra messages too
                        self.output_handler(line)
                # No need to break: the loop should exit when the app is done

            if fn_tracker.any_active():
                log.warning("Some functions didn't transition to 'end'. History: %r", fn_tracker.state_history)

        # 1. Start the app
        # 2. Start consuming logs
        # 3. Yield control to the caller
        # 4. Wait for the logs to finish

        task = None
        self._run_id = str(uuid())[:8]
        try:
            async with self.app.run():
                task = asyncio.create_task(consume(self._run_id))
                yield

        except modal.exception.AuthError as e:
            e.add_note('Tip: Run `./go auth` to authenticate with Modal. If this is a notebook, restart the kernel.')
            raise

        finally:
            self._run_id = None
            if task is not None:
                # Can't wait inside the context manager, because the app would still be running
                try:
                    await asyncio.wait_for(task, timeout=shutdown_timeout)
                except asyncio.TimeoutError:
                    log.warning(f"Output streaming task didn't complete within {shutdown_timeout}s timeout")

        # async with run_with_logs(self.app, log_handler=self.stdout) as log_context:
        #     yield log_context

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
            fn_id = short_id()

            @self.app.function(**kwargs)
            @wraps(fn)
            async def modal_function(run_id, call_id, *_args, **_kwargs):
                # This is called in the remote container

                # These state messages are an integral part of the output streaming
                state = CallState(run_id=run_id, fn_name=fn.__name__, fn_id=fn_id, call_id=call_id, state='guard')
                print(state, flush=True)

                async def guarded_fn() -> R:
                    return await fn(*_args, **_kwargs)

                for guard in reversed(specific_guards):
                    guarded_fn = _wrap_with_guard(guarded_fn, guard, fn)

                for guard in reversed(global_guards):
                    guarded_fn = _wrap_with_guard(guarded_fn, guard, fn)

                state.state = 'start'
                print(state, flush=True)

                try:
                    return await guarded_fn()
                except BaseException as e:
                    state.state = 'error'
                    state.msg = str(e)
                    print(state, flush=True)
                    raise
                finally:
                    state.state = 'end'
                    print(state, flush=True)

            @wraps(fn)
            async def local_wrapper(*args, **kwargs):
                # This is called locally to start the remote function
                if self._run_id is None:
                    raise RuntimeError('Experiment is not running.')
                call_id = short_id()
                return await modal_function.remote.aio(self._run_id, call_id, *args, **kwargs)

            return local_wrapper

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

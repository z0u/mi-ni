from __future__ import annotations

import asyncio
import dataclasses
import os
import typing
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Protocol, runtime_checkable

import modal
from modal._utils.async_utils import TaskContext, synchronize_api
from modal.app import _App
from modal.client import HEARTBEAT_INTERVAL, _Client
from modal.config import config, logger
from modal.exception import InvalidError
from modal.output import _get_output_manager
from modal.runner import _create_all_objects, _heartbeat, _init_local_app_new, _publish_app, _status_based_disconnect
from modal.running_app import RunningApp
from modal_proto import api_pb2


class InterruptHandler(Protocol):
    """mdmd:hidden"""

    @asynccontextmanager
    async def __call__(self) -> AsyncGenerator[None, None]:
        """Handle an interrupt event."""
        yield


@asynccontextmanager
async def _null_interrupt_handler() -> AsyncGenerator[None, None]:
    """mdmd:hidden"""
    yield


@asynccontextmanager
async def _disconnect_on_error(
    client: _Client,
    app_id: str,
    *,
    on_interrupt: Optional[InterruptHandler] = None,
) -> AsyncGenerator[None, None]:
    """Disconnect from the app on any error."""
    if on_interrupt is None:
        on_interrupt = _null_interrupt_handler

    try:
        yield
    except asyncio.CancelledError as e:
        # typically happens on sigint/ctrl-C during setup (the KeyboardInterrupt happens in the main thread)
        async with on_interrupt():
            await _status_based_disconnect(client, app_id, e)
        raise
    except KeyboardInterrupt as e:
        # typically happens if sigint comes in during user code execution
        async with on_interrupt():
            await _status_based_disconnect(client, app_id, e)
        return
    except BaseException as e:
        await _status_based_disconnect(client, app_id, e)
        raise


def _app_state_for(detach: bool):
    """mdmd:hidden"""
    return api_pb2.APP_STATE_DETACHED if detach else api_pb2.APP_STATE_EPHEMERAL


@dataclasses.dataclass
class RunContext:
    running_app: RunningApp
    client: _Client
    environment_name: str
    detach: bool
    task_context: TaskContext

    @property
    def app_id(self) -> str:
        return self.running_app.app_id


@asynccontextmanager
async def _run_app_init(
    app: _App,
    *,
    client: Optional[_Client] = None,
    detach: bool = False,
    environment_name: Optional[str] = None,
    interactive: bool = False,
    grace: Optional[float] = None,
) -> AsyncGenerator[RunContext, None]:
    """Run app, step 1: initialize a running app."""
    if environment_name is None:
        environment_name = typing.cast(str, config.get('environment'))

    if modal._runtime.execution_context._is_currently_importing:
        raise InvalidError('Can not run an app in global scope within a container')

    if app._running_app:
        raise InvalidError(
            "App is already running and can't be started again.\n"
            'You should not use `app.run` or `run_app` within a Modal `local_entrypoint`'
        )

    if app.description is None:
        import __main__

        if '__file__' in dir(__main__):
            app.set_description(os.path.basename(__main__.__file__))
        else:
            # Interactive mode does not have __file__.
            # https://docs.python.org/3/library/__main__.html#import-main
            app.set_description(__main__.__name__)

    if client is None:
        client = await _Client.from_env()

    running_app: RunningApp = await _init_local_app_new(
        client,
        app.description or '',
        environment_name=environment_name or '',
        app_state=_app_state_for(detach),
        interactive=interactive,
    )

    async with app._set_local_app(client, running_app), TaskContext(grace=grace) as tc:
        # Start heartbeats loop to keep the client alive
        # we don't log heartbeat exceptions in detached mode
        # as losing the local connection will not affect the running app
        def heartbeat():
            return _heartbeat(client, running_app.app_id)

        heartbeat_loop = tc.infinite_loop(heartbeat, sleep=HEARTBEAT_INTERVAL, log_exception=not detach)

        yield RunContext(
            running_app=running_app,
            client=client,
            environment_name=environment_name,
            detach=detach,
            task_context=tc,
        )

        # successful completion!
        heartbeat_loop.cancel()
        await _status_based_disconnect(client, running_app.app_id, exc_info=None)


async def _run_app_create(app: _App, ctx: RunContext):
    """Run app, step 2: create objects and publish."""
    await _create_all_objects(ctx.client, ctx.running_app, app._functions, app._classes, ctx.environment_name)
    await _publish_app(ctx.client, ctx.running_app, _app_state_for(ctx.detach), app._functions, app._classes)


@asynccontextmanager
async def _run_app(
    app: _App,
    *,
    client: Optional[_Client] = None,
    detach: bool = False,
    environment_name: Optional[str] = None,
    interactive: bool = False,
) -> AsyncGenerator[_App, None]:
    """mdmd:hidden"""
    output_mgr = _get_output_manager()
    if interactive and output_mgr is None:
        msg = 'Interactive mode requires output to be enabled. (Use the the `modal.enable_output()` context manager.)'
        raise InvalidError(msg)

    logs_timeout = config['logs_timeout']

    async with _run_app_init(
        app,
        client=client,
        detach=detach,
        environment_name=environment_name,
        interactive=interactive,
        grace=logs_timeout,
    ) as ctx:
        logs_loop: Optional[asyncio.Task] = None
        running_app = ctx.running_app

        if output_mgr is not None:
            # Defer import so this module is rich-safe
            # TODO(michael): The get_app_logs_loop function is itself rich-safe aside from accepting an OutputManager
            # as an argument, so with some refactoring we could avoid the need for this deferred import.
            from modal._output import get_app_logs_loop

            with output_mgr.make_live(output_mgr.step_progress('Initializing...')):
                initialized_msg = (
                    f'Initialized. [grey70]View run at [underline]{running_app.app_page_url}[/underline][/grey70]'
                )
                output_mgr.print(output_mgr.step_completed(initialized_msg))
                output_mgr.update_app_page_url(running_app.app_page_url or 'ERROR:NO_APP_PAGE')

            # Start logs loop

            logs_loop = ctx.task_context.create_task(
                get_app_logs_loop(ctx.client, output_mgr, app_id=ctx.app_id, app_logs_url=running_app.app_logs_url)
            )

        @asynccontextmanager
        async def on_cancel():
            # this typically happens on sigint/ctrl-C during setup (the KeyboardInterrupt happens in the main thread)
            if output_mgr := _get_output_manager():
                output_mgr.print('Aborting app initialization...\n')
            yield  # Allow disconnect to happen

        async with _disconnect_on_error(ctx.client, ctx.app_id, on_interrupt=on_cancel):
            await _run_app_create(app, ctx)

        if detach:

            @asynccontextmanager
            async def on_interrupt():
                if output_mgr := _get_output_manager():
                    output_mgr.print(output_mgr.step_completed('Shutting down Modal client.'))
                    output_mgr.print(
                        'The detached app keeps running. You can track its progress at: '
                        f'[magenta]{running_app.app_page_url}[/magenta]'
                        ''
                    )
                if logs_loop:
                    logs_loop.cancel()
                yield  # Allow disconnect to happen

        else:

            @asynccontextmanager
            async def on_interrupt():
                if output_mgr := _get_output_manager():
                    output_mgr.print(
                        'Disconnecting from Modal - This will terminate your Modal app in a few seconds.\n'
                    )
                yield  # Allow disconnect to happen
                if logs_loop:
                    try:
                        await asyncio.wait_for(logs_loop, timeout=logs_timeout)
                    except asyncio.TimeoutError:
                        logger.warning('Timed out waiting for final app logs.')

                if output_mgr:
                    output_mgr.print(
                        output_mgr.step_completed(
                            'App aborted. '
                            f'[grey70]View run at [underline]{running_app.app_page_url}[/underline][/grey70]'
                        )
                    )

        async with _disconnect_on_error(ctx.client, ctx.app_id, on_interrupt=on_interrupt):
            # Show logs from dynamically created images.
            # TODO: better way to do this
            if output_mgr := _get_output_manager():
                output_mgr.enable_image_logs()

            # Yield to context
            if output_mgr := _get_output_manager():
                with output_mgr.show_status_spinner():
                    yield app
            else:
                yield app

        # wait for logs gracefully, even though the task context would do the same
        # this allows us to log a more specific warning in case the app doesn't
        # provide all logs before exit
        if logs_loop:
            try:
                await asyncio.wait_for(logs_loop, timeout=logs_timeout)
            except asyncio.TimeoutError:
                logger.warning('Timed out waiting for final app logs.')

    if output_mgr := _get_output_manager():
        output_mgr.print(
            output_mgr.step_completed(
                f'App completed. [grey70]View run at [underline]{running_app.app_page_url}[/underline][/grey70]'
            )
        )


@runtime_checkable
class RunEventHandler(Protocol):
    async def on_init(self, app: _App, ctx: RunContext) -> None:
        """Handle the creation of a run context."""
        pass

    async def on_create(self, app: _App, ctx: RunContext) -> None:
        """Handle actions before the app is created."""
        pass

    @asynccontextmanager
    async def on_interrupt(self) -> AsyncGenerator[None, None]:
        """Handle an interrupt event."""
        yield


@asynccontextmanager
async def _run_app_silent(
    app: _App,
    *,
    handler: RunEventHandler,
    client: Optional[_Client] = None,
    detach: bool = False,
    environment_name: Optional[str] = None,
) -> AsyncGenerator[_App, None]:
    grace = config['logs_timeout']

    async with _run_app_init(app, client=client, detach=detach, environment_name=environment_name, grace=grace) as ctx:
        await handler.on_init(app, ctx)
        async with _disconnect_on_error(ctx.client, ctx.app_id, on_interrupt=handler.on_interrupt):
            await _run_app_create(app, ctx)
            await handler.on_create(app, ctx)
            yield app


run_app = synchronize_api(_run_app)

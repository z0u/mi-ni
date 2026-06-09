"""
Apparatus for running sweeps on Modal infrastructure.

Example::

    from mini.modal_apparatus import ModalApparatus

    app = ModalApparatus("my-experiment").w(gpu="T4", timeout=3600)
    results = list(app.map(train, configs))
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager, nullcontext
from functools import wraps
from itertools import count
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable, TypeVar, override

import modal

from mini._queues import QueueLike
from mini.apparatus import Apparatus
from mini.modal_queue import ModalQueue
from mini.modal_volume import ModalVolume
from mini.progress import ProgressMessage, progress_context
from mini.progress_display import RichProgressDisplay
from mini.requirements import project_packages, strip_build_tags, uv_freeze
from mini.volume import data_dir_context

log = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

__all__ = ['ModalApparatus']

STARTUP_TIMEOUT_SECONDS = 120


def _modal_auth_error_message() -> str:
    """Build a user-facing message for Modal authentication failures."""
    return 'Modal authentication failed. Run ./go auth, then try again.'


def _app_page_url(app: modal.App) -> str | None:
    """Extract the dashboard URL from a running Modal app.

    The URL comes from the backend via ``RunningApp.app_page_url``, but
    synchronicity hides ``_running_app`` on the public ``App`` wrapper.
    Reach through the wrapper to get it.
    """
    # synchronicity stores the underlying _App as the sole entry in __dict__
    inner_values = list(app.__dict__.values())
    if not inner_values:
        return None
    inner_app = inner_values[0]
    running = getattr(inner_app, '_running_app', None)
    if running is None:
        return None
    return running.app_page_url


def make_image() -> modal.Image:
    """Helper to create a Modal image with experiment dependencies."""
    deps = uv_freeze(all_groups=True, not_groups=['local', 'dev'])
    # Remove build tags (e.g., +cpu, +cu121) to improve cross-platform compatibility. This allows Torch to use devices available in the Modal environment.
    generic_deps = strip_build_tags(deps)
    project_deps = project_packages()
    print(f'Creating Modal image with dependencies: Project: {project_deps}')
    return (
        modal.Image.debian_slim()
        .pip_install(*generic_deps)
        .add_local_python_source(*project_deps)
    )  # fmt: skip


class ModalApparatus(Apparatus[ModalVolume]):
    """
    Run functions on Modal.

    Usage::

        app = ModalApparatus("my-experiment").w(gpu="T4", timeout=3600)
        results = list(app.map(train, configs))
    """

    app: modal.App

    def __init__(self, app: modal.App | str):
        if isinstance(app, str):
            name = app
            self.app = modal.App(name)
        else:
            if not app.name:
                raise ValueError('ModalApparatus requires a named modal.App')
            name = app.name
            self.app = app
        self.modal_fn_kwargs: dict[str, Any] = {
            'image': make_image(),
            'max_containers': 1,
            # Don't let Modal silently retry failures — surface them immediately.
            'retries': 0,
        }
        self._before_hooks: list[Callable[[], Any]] = []
        self._volume: ModalVolume | None = ModalVolume(name)

    def __str__(self) -> str:
        return f'Modal apparatus "{self.app.name}"'

    def clone(self) -> ModalApparatus:
        new_app = ModalApparatus(self.app)
        new_app.modal_fn_kwargs = self.modal_fn_kwargs.copy()
        new_app._before_hooks = self._before_hooks[:]
        new_app._volume = self._volume
        return new_app

    def w(self, **kwargs: Any) -> ModalApparatus:
        """
        Return a new apparatus with additional Modal function kwargs merged in.

        These kwargs are passed to the ``@app.function()`` decorator when
        mapping, and can be used to specify things like GPU requirements or
        timeouts.
        """
        new_app = self.clone()
        new_app.modal_fn_kwargs = {**self.modal_fn_kwargs, **kwargs}
        return new_app

    @override
    def before_each(self, hook: Callable[[], Any]) -> ModalApparatus:
        new_app = self.clone()
        new_app._before_hooks = self._before_hooks + [hook]
        return new_app

    @override
    async def amap(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ):
        try:
            async for result in self._amap(fn, *iterables, kwargs=kwargs):
                yield result
        except modal.exception.AuthError:
            log.debug('Modal authentication failed', exc_info=True)
            raise RuntimeError(_modal_auth_error_message()) from None

    async def _amap(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ):
        # TODO: support lazy iterables
        iterables_lists: list[list] = [list(it) for it in iterables]
        n = len(iterables_lists[0]) if iterables_lists else 0
        if n == 0:
            return

        log.info('Running %d jobs on Modal', n)
        run_id = secrets.token_hex(4)

        image: modal.Image = self.modal_fn_kwargs.get('image') or modal.Image.debian_slim()
        with modal.enable_output():
            async with self.app.run():
                await image.build.aio(self.app)

        async with modal.Queue.ephemeral() as progress_queue:
            display = RichProgressDisplay(total_jobs=n, queue=ModalQueue(progress_queue))
            modal_fn, startup_timeout = self._build_modal_fn(
                fn,
                run_id,
                display,
                kwargs=kwargs,
            )

            async with display, self.app.run():
                if url := _app_page_url(self.app):
                    print(f'View app at {url}')
                async with _startup_watchdog(display, startup_timeout):
                    async for result in modal_fn.map.aio(count(), *iterables_lists):
                        yield result

    def _build_modal_fn(
        self,
        fn: Callable[..., R],
        run_id: str,
        display: RichProgressDisplay,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[modal.Function, float]:
        """Wrap *fn* for Modal and register it with the app.

        Return ``(modal_function, startup_timeout)``.
        """
        max_containers = self.modal_fn_kwargs.get('max_containers', 1)
        emission_interval = max_containers / 10.0
        wrapped_fn = _wrap_for_modal(
            fn,
            self._before_hooks,
            run_id,
            queue=display.queue,
            kwargs=kwargs or {},
            emission_interval=emission_interval,
            data_dir=self._volume.path if self._volume is not None else None,
            commit_volume=(self._volume._modal_volume if isinstance(self._volume, ModalVolume) else None),
        )
        fn_kwargs = {**self.modal_fn_kwargs}
        startup_timeout: float = fn_kwargs.pop('startup_timeout', STARTUP_TIMEOUT_SECONDS)
        if isinstance(self._volume, ModalVolume):
            volumes = fn_kwargs.get('volumes', {})
            fn_kwargs['volumes'] = {
                **volumes,
                str(self._volume.path): self._volume._modal_volume,
            }
        modal_fn = self.app.function(serialized=True, **fn_kwargs)(wrapped_fn)
        return modal_fn, startup_timeout


@asynccontextmanager
async def _startup_watchdog(
    display: RichProgressDisplay,
    timeout_seconds: float,
) -> AsyncIterator[None]:
    """Raise if no remote container checks in within *timeout_seconds*.

    Once the display receives any message (set via ``display._any_message``),
    the deadline is cancelled and the body runs without a time limit.
    """
    try:
        async with asyncio.timeout(timeout_seconds) as scope:

            async def _cancel_on_first_message() -> None:
                await asyncio.to_thread(
                    display._any_message.wait,
                    timeout_seconds + 10,
                )
                scope.reschedule(None)

            watcher = asyncio.create_task(_cancel_on_first_message())
            try:
                yield
            finally:
                watcher.cancel()
    except TimeoutError:
        raise RuntimeError(
            f'No containers started within {timeout_seconds}s. '
            'Containers may be crash-looping — '
            'check the Modal dashboard for logs.'
        ) from None


def _wrap_for_modal(
    fn: Callable[..., R],
    hooks: list[Callable[[], None]],
    run_id: str,
    queue: QueueLike[ProgressMessage],
    kwargs: dict[str, Any],
    emission_interval: float,
    data_dir: Path | None,
    commit_volume: modal.Volume | None = None,
) -> Callable[..., R]:
    @wraps(fn)
    def wrapped_fn(index: int, *args) -> R:
        # Signal that this container started successfully. Emitted directly
        # (not via the debouncer) so the caller-side watchdog sees it ASAP.
        queue.put(
            ProgressMessage(
                run_id=run_id,
                job_id=str(index),
                step=0,
                total=0,
                message='started',
            )
        )
        dir_ctx = data_dir_context(data_dir) if data_dir is not None else nullcontext()
        with progress_context(run_id, str(index), queue=queue, emission_interval=emission_interval), dir_ctx:
            for hook in reversed(hooks):
                hook()
            result = fn(*args, **kwargs)
            if commit_volume is not None:
                commit_volume.commit()
            return result

    # Give the wrapper a unique name so that repeated submissions of the same
    # function on a single App don't trigger Modal's name-collision warning.
    wrapped_fn.__name__ = f'{wrapped_fn.__name__}_{run_id}'
    wrapped_fn.__qualname__ = f'{wrapped_fn.__qualname__}_{run_id}'

    return wrapped_fn

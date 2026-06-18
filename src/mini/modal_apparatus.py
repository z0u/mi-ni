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
import time
from contextlib import asynccontextmanager, nullcontext
from functools import wraps
from itertools import count
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable, TypeVar, override

import cloudpickle
import modal

from mini._queues import QueueLike
from mini.apparatus import Apparatus
from mini.memo import MemoStore, RecordStore
from mini.modal_queue import ModalQueue
from mini.modal_volume import ModalVolume
from mini.progress import ProgressMessage, progress_context
from mini.progress_display import RichProgressDisplay
from mini.requirements import project_packages, uv_freeze
from mini.volume import data_dir_context

log = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

__all__ = ['ModalApparatus', 'ModalRecordStore', 'ModalMemoStore']

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
    """Helper to create a Modal image with experiment dependencies.

    Includes the `cuda` dependency group (e.g. `jax[cuda12]`), which is excluded
    from local installs: locally we run CPU-only, while the remote image gets
    the CUDA plugin and picks up the GPU when one is attached.
    """
    deps = uv_freeze(all_groups=True, not_groups=['local', 'dev'])
    project_deps = project_packages()
    print(f'Creating Modal image with dependencies: Project: {project_deps}')
    return (
        modal.Image.debian_slim()
        .pip_install(*deps)
        .add_local_python_source(*project_deps)
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Memoized-orchestration backend (control plane = modal.Dict, I/O plane = Volume)
# ---------------------------------------------------------------------------


class ModalRecordStore(RecordStore):
    """``RecordStore`` backed by a named ``modal.Dict``.

    The Dict is readable/writable from the client with no remote function and no
    commit (Redis-backed), so polling never spins up compute. The same named
    Dict is opened by the remote worker to write back state/metrics. Records are
    tiny and last-writer-wins, so a Dict value per key is the natural fit.
    """

    def __init__(self, d: Any):
        # *d* is a ``modal.Dict`` (or any get/keys/__setitem__ mapping — a plain
        # dict for tests). Injected rather than opened here so the store is
        # testable without the network; use ``from_name`` for the real thing.
        self._d = d

    @classmethod
    def from_name(cls, name: str) -> ModalRecordStore:
        return cls(modal.Dict.from_name(name, create_if_missing=True))

    def read(self, key: str) -> dict[str, Any] | None:
        return self._d.get(key)

    def write(self, key: str, record: dict[str, Any]) -> None:
        self._d[key] = record

    def merge(self, key: str, fields: dict[str, Any]) -> None:
        cur = self._d.get(key) or {}
        cur.update(fields)
        self._d[key] = cur

    def keys(self) -> list[str]:
        return list(self._d.keys())


class ModalMemoStore(MemoStore):
    """A ``MemoStore`` whose records live in a ``modal.Dict`` and whose results
    are read back from the Modal Volume (the remote worker writes them there and
    commits). Only the I/O-plane *reads* differ from the local store — the remote
    worker, with the Volume mounted, writes through a plain ``MemoStore``.
    """

    def __init__(self, volume: ModalVolume, records: RecordStore):
        super().__init__(volume.path, records=records)
        self._volume = volume

    def _read_volume_bytes(self, rel: str) -> bytes:
        # Client-side reads already reflect the worker's committed writes;
        # ``reload()`` is only valid inside a running function, not here.
        return b''.join(self._volume._modal_volume.read_file(rel))

    def result(self, key: str) -> Any:
        return cloudpickle.loads(self._read_volume_bytes(f'_memo/{key}/result.pkl'))

    def error(self, key: str) -> str:
        try:
            return self._read_volume_bytes(f'_memo/{key}/error.txt').decode()
        except FileNotFoundError, modal.exception.NotFoundError:
            return '(no logs)'


def _modal_task_entry(blob: bytes, key: str, dict_name: str, volume_name: str, mount_point: str) -> None:
    """Remote entry: run one memoized call on Modal and persist its result/state.

    Mirrors the local subprocess worker (``mini._taskworker``) but reads the call
    from the ``spawn`` argument (not disk), writes records to the ``modal.Dict``,
    and commits the Volume before flipping the record to a settled state.
    """
    import cloudpickle as _cp
    import modal as _modal

    from mini._taskworker import execute_task
    from mini.memo import MemoStore
    from mini.modal_apparatus import ModalRecordStore

    fn, args, hooks = _cp.loads(blob)
    store = MemoStore(Path(mount_point), records=ModalRecordStore.from_name(dict_name))
    volume = _modal.Volume.from_name(volume_name)
    execute_task(store, key, fn, args, hooks, commit=volume.commit)


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
        self._memo_fn: modal.Function | None = None

    def __str__(self) -> str:
        return f'Modal apparatus "{self.app.name}"'

    def clone(self) -> ModalApparatus:
        new_app = ModalApparatus(self.app)
        new_app.modal_fn_kwargs = self.modal_fn_kwargs.copy()
        new_app._before_hooks = self._before_hooks[:]
        new_app._volume = self._volume
        return new_app

    @property
    def _dict_name(self) -> str:
        """Name of the control-plane ``modal.Dict`` for this experiment."""
        return f'mini-cp-{self.app.name}'

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

    # -- Memoized orchestration (detached) ------------------------------------

    @override
    def memo_store(self) -> MemoStore:
        return ModalMemoStore(self.volume, ModalRecordStore.from_name(self._dict_name))

    def _memo_worker(self) -> modal.Function:
        """Register (once) and return the generic remote worker for memo tasks.

        One stable function serves every task; each spawned call carries its own
        cloudpickled call. The Volume is mounted so the worker writes results to
        the same path the client reads back from.

        The ``max_containers=1`` default (sensible for the blocking ``amap``)
        would serialise a detached sweep through one container, so it's dropped
        here — a fanned-out ``ctx.map`` should parallelise. Pass an explicit
        ``.w(max_containers=N)`` to cap concurrency.
        """
        if self._memo_fn is None:
            drop = {'startup_timeout', 'max_containers'}
            fn_kwargs = {k: v for k, v in self.modal_fn_kwargs.items() if k not in drop}
            if isinstance(self._volume, ModalVolume):
                fn_kwargs['volumes'] = {
                    **fn_kwargs.get('volumes', {}),
                    str(self._volume.path): self._volume._modal_volume,
                }
            self._memo_fn = self.app.function(serialized=True, **fn_kwargs)(_modal_task_entry)
        return self._memo_fn

    @override
    def spawn_tasks(self, store: MemoStore, batch: list[tuple[str, Callable, tuple, list]]) -> None:
        worker = self._memo_worker()
        blobs = [cloudpickle.dumps((fn, args, hooks)) for _, fn, args, hooks in batch]
        keys = [key for key, *_ in batch]
        # One detached spawn_map for the whole batch, rather than an app.run per
        # task. The workers outlive this block (and this process); a later wake
        # polls their state from the Dict. NB on Modal 1.3.x spawn_map returns
        # None (no FunctionCall to poll), so liveness rests on Dict heartbeats.
        with self.app.run(detach=True):
            worker.spawn_map(
                blobs,
                keys,
                kwargs={
                    'dict_name': self._dict_name,
                    'volume_name': self.app.name,
                    'mount_point': str(self.volume.path),
                },
            )
        now = time.time()
        for key in keys:
            store.update(key, heartbeat_at=now)

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

"""
Apparatus for running sweeps on Modal infrastructure.

Example::

    from mini.modal_apparatus import ModalApparatus

    app = ModalApparatus("my-experiment").w(gpu="T4", timeout=3600)
    results = list(app.map(train, configs))
"""

from __future__ import annotations

import logging
import secrets
from collections import deque
from contextlib import nullcontext
from itertools import count
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Iterable, TypeVar, cast, override

import modal

from mini._queues import EndOfQueue, QueueLike
from mini.apparatus import Apparatus
from mini.modal_volume import ModalVolume
from mini.progress import ProgressMessage, progress_context
from mini.progress_display import RichProgressDisplay
from mini.requirements import project_packages, strip_build_tags, uv_freeze
from mini.volume import data_dir_context

log = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

__all__ = ['ModalApparatus']


def make_image() -> modal.Image:
    """Helper to create a Modal image with experiment dependencies."""
    deps = uv_freeze(all_groups=True, not_groups='local')
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

    Functions can report progress via stdout (using print) or by calling
    ``emit_progress()``, which emits URN-formatted progress messages. When
    ``show_progress=True`` and running in a terminal, the executor shows
    job-level completion in a Rich progress display.

    Usage::

        app = ModalApparatus("my-experiment", show_progress=True).w(gpu="T4", timeout=3600)
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
        }
        self._before_hooks: list[Callable[[], Any]] = []
        self._volume: ModalVolume | None = ModalVolume(name)

    def __str__(self) -> str:
        return f'Modal apparatus "{self.app.name}"'

    def clone(self) -> ModalApparatus:
        new_executor = ModalApparatus(self.app)
        new_executor.modal_fn_kwargs = self.modal_fn_kwargs.copy()
        new_executor._before_hooks = self._before_hooks[:]
        new_executor._volume = self._volume
        return new_executor

    def w(self, **kwargs: Any) -> ModalApparatus:
        """
        Return a new executor with additional Modal function kwargs merged in.

        These kwargs are passed to the ``@app.function()`` decorator when
        mapping, and can be used to specify things like GPU requirements or
        timeouts.
        """
        new_executor = self.clone()
        new_executor.modal_fn_kwargs = {**self.modal_fn_kwargs, **kwargs}
        return new_executor

    @override
    def before_each(self, hook: Callable[[], Any]) -> ModalApparatus:
        new_executor = self.clone()
        new_executor._before_hooks = self._before_hooks + [hook]
        return new_executor

    @override
    async def amap(
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

        hooks = self._before_hooks

        image: modal.Image = self.modal_fn_kwargs.get('image') or modal.Image.debian_slim()
        with modal.enable_output(), self.app.run():
            image.build(self.app)

        with modal.Queue.ephemeral() as progress_queue:
            progress_display = RichProgressDisplay(total_jobs=n, queue=ModalQueue(progress_queue))
            # Target ~10 emissions/sec overall: interval = max_containers / target_rate_hz
            max_containers = self.modal_fn_kwargs.get('max_containers', 1)
            emission_interval = max_containers / 10.0
            wrapped_fn = _wrap_for_modal(
                fn,
                hooks,
                run_id,
                queue=progress_display.queue,
                kwargs=kwargs or {},
                emission_interval=emission_interval,
                data_dir=self._volume.path if self._volume is not None else None,
                commit_volume=(self._volume._modal_volume if isinstance(self._volume, ModalVolume) else None),
            )
            # The `function` decorator must be applied *before* `app.run()` starts the app.
            fn_kwargs = {**self.modal_fn_kwargs}
            if isinstance(self._volume, ModalVolume):
                volumes = fn_kwargs.get('volumes', {})
                fn_kwargs['volumes'] = {**volumes, str(self._volume.path): self._volume._modal_volume}
            modal_fn = self.app.function(serialized=True, **fn_kwargs)(wrapped_fn)

            with progress_display, self.app.run():
                async for result in modal_fn.map.aio(count(), *iterables_lists):
                    yield result


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
    # @wraps(fn)  # Don't use wraps: it confuses Modal
    def wrapped_fn(index: int, *args) -> R:
        dir_ctx = data_dir_context(data_dir) if data_dir is not None else nullcontext()
        with progress_context(run_id, str(index), queue=queue, emission_interval=emission_interval), dir_ctx:
            for hook in reversed(hooks):
                hook()
            result = fn(*args, **kwargs)
            if commit_volume is not None:
                commit_volume.commit()
            return result

    return wrapped_fn


class ModalQueue(QueueLike[T]):
    """A Modal-backed queue with buffered batch reads."""

    def __init__(self, queue: modal.Queue, batch_size: int = 5_000):
        self._queue = queue
        self._batch_size = batch_size
        self._buffer: deque[T] = deque()
        self._saw_end = False

    def put(self, item: T | EndOfQueue, /, block: bool = True, timeout: float | None = None) -> None:
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, /, block: bool = True, timeout: float | None = None) -> T:
        if self._buffer:
            return self._buffer.popleft()
        if self._saw_end:
            raise EndOfQueue()

        # Modal's Queue returns None instead of raising Empty when no item is available.
        items = self._queue.get_many(self._batch_size, block=block, timeout=timeout)
        if not items:
            raise Empty('Modal queue returned no items, treating as empty')

        cleaned: list[T] = []
        for item in items:
            if isinstance(item, EndOfQueue):
                self._saw_end = True
                break
            if item is None:
                continue
            cleaned.append(cast(T, item))

        if not cleaned:
            if self._saw_end:
                raise EndOfQueue()
            raise Empty('Modal queue returned no items, treating as empty')

        self._buffer.extend(cleaned)
        return self._buffer.popleft()

    def empty(self) -> bool:
        # Modal's Queue doesn't have an empty() method.
        return self._queue.len() == 0

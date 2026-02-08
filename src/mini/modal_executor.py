"""
Modal executor for running experiment sweeps on Modal infrastructure.

Example::

    import modal
    from mini.modal_executor import ModalExecutor

    app = modal.App("my-experiment")
    executor = ModalExecutor(app).with_modal_kwargs(gpu="T4", timeout=3600)
    results = list(executor.map(train, configs))
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections import deque
from itertools import count
from queue import Empty
from typing import Any, Callable, Iterable, TypeVar, cast, override

import modal

from mini._queues import EndOfQueue, QueueLike
from mini.executor import Executor
from mini.progress import ProgressMessage, progress_context
from mini.progress_display import RichProgressDisplay
from utils.requirements import freeze, project_packages

log = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

__all__ = ['ModalExecutor']


def _is_async_context() -> bool:
    """Check if we're running inside an async context (e.g., Marimo, Jupyter with async)."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def make_app(name: str) -> modal.App:
    """Helper to create a Modal app."""
    return modal.App(name)


def make_image() -> modal.Image:
    """Helper to create a Modal image with experiment dependencies."""
    return (
        modal.Image.debian_slim()
        .pip_install(*freeze(all=True, local=False))
        .add_local_python_source(*project_packages())
    )  # fmt: skip


class ModalExecutor(Executor):
    """
    Run functions on Modal.

    Functions can report progress via stdout (using print) or by calling
    ``emit_progress()``, which emits URN-formatted progress messages. When
    ``show_progress=True`` and running in a terminal, the executor shows
    job-level completion in a Rich progress display.

    Usage::

        executor = ModalExecutor("my-experiment", show_progress=True).w(gpu="T4", timeout=3600)
        results = list(executor.map(train, configs))
    """

    app: modal.App

    def __init__(self, app: modal.App | str):
        self.app = modal.App(app) if isinstance(app, str) else app
        self.modal_fn_kwargs: dict[str, Any] = {
            'image': make_image(),
            'max_containers': 1,
        }
        self._before_hooks: list[Callable[[], None]] = []

    def clone(self) -> ModalExecutor:
        new_executor = ModalExecutor(self.app)
        new_executor.modal_fn_kwargs = self.modal_fn_kwargs.copy()
        new_executor._before_hooks = self._before_hooks[:]
        return new_executor

    def w(self, **kwargs: Any) -> ModalExecutor:
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
    def before_each(self, hook: Callable[[], None]) -> ModalExecutor:
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

        log.info('[ModalExecutor] Running %d jobs on Modal', n)
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
            )
            # The `function` decorator must be applied *before* `app.run()` starts the app.
            modal_fn = self.app.function(serialized=True, **self.modal_fn_kwargs)(wrapped_fn)

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
) -> Callable[..., R]:
    # @wraps(fn)  # Don't use wraps: it confuses Modal
    def wrapped_fn(index: int, *args) -> R:
        with progress_context(run_id, str(index), queue=queue, emission_interval=emission_interval):
            for hook in reversed(hooks):
                hook()
            return fn(*args, **kwargs)

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

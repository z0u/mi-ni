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
import queue as queue_module
import secrets
import threading
from itertools import count
from queue import Empty
from typing import Any, Callable, Iterable, Iterator, TypeVar, cast, override

import modal

from mini._queues import EndOfQueue, QueueLike
from mini.executor import Executor
from mini.progress import ProgressMessage, reset_job_context, set_job_context
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


def _run_async_map_in_thread(modal_fn, *args, **kwargs) -> Iterator[Any]:
    """
    Run Modal's async map in a separate thread to avoid nested event loop issues.

    This is needed when the executor is used in async contexts like Marimo,
    which run cells in an async event loop. Modal's synchronous iteration
    doesn't work in that case, so we run the async version in its own thread.
    """
    results_queue: queue_module.Queue = queue_module.Queue()
    exception_holder: list[Exception] = []

    def run_in_thread():
        try:
            async def collect():
                async for result in modal_fn.map.aio(*args, **kwargs):
                    results_queue.put(('result', result))
                results_queue.put(('done', None))

            # Create a new event loop for this thread
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

    # Wait for thread to finish and check for exceptions
    thread.join(timeout=1.0)
    if exception_holder:
        raise exception_holder[0]


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
    def map(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> Iterator[R]:
        iterables_lists: list[list] = [list(it) for it in iterables]
        n = len(iterables_lists[0]) if iterables_lists else 0
        if n == 0:
            return

        log.info('[ModalExecutor] Running %d jobs on Modal', n)
        run_id = secrets.token_hex(4)

        hooks = self._before_hooks

        image = self.modal_fn_kwargs.get('image') or modal.Image.debian_slim()
        with modal.enable_output(), self.app.run():
            image.build(self.app)

        with modal.Queue.ephemeral() as progress_queue:
            progress_display = RichProgressDisplay(total_jobs=n, queue=ModalQueue(progress_queue))
            wrapped_fn = _wrap_for_modal(fn, hooks, run_id, queue=progress_display.queue, kwargs=kwargs or {})
            # The `function` decorator must be applied *before* `app.run()` starts the app.
            modal_fn = self.app.function(serialized=True, **self.modal_fn_kwargs)(wrapped_fn)

            with progress_display, self.app.run():
                # Check if we're in an async context (e.g., Marimo)
                # If so, use async API in a separate thread to avoid nested event loop issues
                if _is_async_context():
                    log.info('[ModalExecutor] Detected async context, using threaded async map')
                    yield from _run_async_map_in_thread(modal_fn, count(), *iterables_lists)
                else:
                    yield from modal_fn.map(count(), *iterables_lists)


def _wrap_for_modal(
    fn: Callable[..., R],
    hooks: list[Callable[[], None]],
    run_id: str,
    queue: QueueLike[ProgressMessage],
    kwargs: dict[str, Any],
) -> Callable[..., R]:
    # @wraps(fn)  # Don't use wraps: it confuses Modal
    def wrapped_fn(index: int, *args) -> R:
        tok1, tok2, tok3 = set_job_context(run_id, str(index), queue=queue)
        try:
            for hook in reversed(hooks):
                hook()
            return fn(*args, **kwargs)
        finally:
            reset_job_context(tok1, tok2, tok3)

    return wrapped_fn


class ModalQueue(QueueLike[T]):
    """A simple thread-safe queue for local use."""

    def __init__(self, queue: modal.Queue):
        self._queue = queue

    def put(self, item: T | EndOfQueue, /, block: bool = True, timeout: float | None = None) -> None:
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, /, block: bool = True, timeout: float | None = None) -> T:
        # Modal's Queue returns None instead of raising Empty when no item is available.
        item = self._queue.get(block=block, timeout=timeout)
        if isinstance(item, EndOfQueue):
            raise item
        if item is None:
            raise Empty('Modal queue returned None, treating as empty')
        return cast(T, item)

    def empty(self) -> bool:
        # Modal's Queue doesn't have an empty() method.
        return self._queue.len() == 0

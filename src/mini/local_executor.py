"""
Local executor for running experiment sweeps with thread-based concurrency.

Example::

    from mini.local_executor import LocalExecutor

    executor = LocalExecutor("my-experiment", max_workers=4)

    # Progress bars are shown automatically when running in a terminal
    results = list(executor.map(train, configs))

    # Disable progress display if needed
    executor = LocalExecutor("my-experiment", max_workers=4, show_progress=False)
"""

from __future__ import annotations

import logging
import secrets
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from queue import Queue
from typing import Any, Callable, Iterable, Iterator, Sized, TypeVar, override

from mini._queues import EndOfQueue, QueueLike
from mini.executor import Executor
from mini.progress import ProgressMessage, reset_job_context, set_job_context
from mini.progress_display import RichProgressDisplay

log = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

__all__ = ['LocalExecutor']


class LocalExecutor(Executor):
    """
    Run functions locally using a thread pool.

    Jobs can report progress via ``emit_progress()`` which is automatically
    displayed using Rich progress bars when running in a terminal.
    """

    def __init__(self, name: str, max_workers: int = 1):
        self.name = name
        self.max_workers = max_workers
        self._before_hooks: list[Callable[[], None]] = []

    def clone(self) -> LocalExecutor:
        new_executor = LocalExecutor(self.name, self.max_workers)
        new_executor._before_hooks = self._before_hooks[:]
        return new_executor

    @override
    def before_each(self, hook: Callable[[], None]) -> LocalExecutor:
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
        sizes = [len(it) for it in iterables if isinstance(it, Sized)]
        n = min(sizes) if sizes else None

        log.info('[%s] Running %d jobs with %d workers', self.name, n, self.max_workers)
        run_id = secrets.token_hex(4)

        progress_display = RichProgressDisplay(n or 0, queue=LocalQueue())
        local_fn = _wrap_for_local(fn, self._before_hooks, run_id, progress_display.queue, kwargs=kwargs or {})

        progress_display.start()
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                yield from pool.map(local_fn, count(), *iterables)
        finally:
            progress_display.stop()


def _wrap_for_local(
    fn: Callable[..., R],
    hooks: list[Callable[[], None]],
    run_id: str,
    queue: QueueLike[ProgressMessage],
    kwargs: dict[str, Any],
) -> Callable[..., R]:
    def run_one(index: int, *args) -> R:
        tok1, tok2, tok3 = set_job_context(run_id, str(index), queue=queue)
        try:
            for hook in reversed(hooks):
                hook()
            result = fn(*args, **kwargs)
            return result
        finally:
            reset_job_context(tok1, tok2, tok3)

    return run_one


class LocalQueue(QueueLike[T]):
    """A simple thread-safe queue for local use."""

    def __init__(self):
        self._queue: Queue[T | EndOfQueue] = Queue()

    def put(self, item: T | EndOfQueue, /, block: bool = True, timeout: float | None = None) -> None:
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, /, block: bool = True, timeout: float | None = None) -> T:
        item = self._queue.get(block=block, timeout=timeout)
        if isinstance(item, EndOfQueue):
            raise item
        return item

    def empty(self) -> bool:
        return self._queue.empty()

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

import asyncio
import logging
import secrets
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, AsyncGenerator, Callable, Iterable, TypeVar, override

from mini._queues import EndOfQueue, QueueLike
from mini.executor import Executor
from mini.progress import ProgressMessage, progress_context
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
    async def amap(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[R, None]:
        # TODO: support lazy iterables
        iterables_lists: list[list] = [list(it) for it in iterables]
        sizes = [len(it) for it in iterables_lists]
        n = min(sizes) if sizes else None

        log.info('[%s] Running %d jobs with %d workers', self.name, n, self.max_workers)
        run_id = secrets.token_hex(4)

        progress_display = RichProgressDisplay(n or 0, queue=LocalQueue())
        # Target ~10 emissions/sec overall: interval = max_workers / target_rate_hz
        emission_interval = self.max_workers / 10.0
        local_fn = _wrap_for_local(
            fn,
            self._before_hooks,
            run_id,
            progress_display.queue,
            kwargs=kwargs or {},
            emission_interval=emission_interval,
        )

        loop = asyncio.get_running_loop()

        with progress_display, ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # Submit all tasks
            tasks = [
                loop.run_in_executor(pool, local_fn, i, *args)
                for i, args in enumerate(zip(*iterables_lists, strict=False))
            ]

            # Yield results in input order to match map semantics
            for task in tasks:
                yield await task


def _wrap_for_local(
    fn: Callable[..., R],
    hooks: list[Callable[[], None]],
    run_id: str,
    queue: QueueLike[ProgressMessage],
    kwargs: dict[str, Any],
    emission_interval: float,
) -> Callable[..., R]:
    def run_one(index: int, *args) -> R:
        with progress_context(run_id, str(index), queue=queue, emission_interval=emission_interval):
            for hook in reversed(hooks):
                hook()
            result = fn(*args, **kwargs)
            return result

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

"""
Local executor for running experiment sweeps with thread-based concurrency.

Example::

    from mini.local_executor import LocalExecutor

    executor = LocalExecutor("my-experiment", max_workers=4)
    results = list(executor.map(train, [{"seed": 1}, {"seed": 2}]))
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Iterable, Iterator, TypeVar

from mini.executor import ProgressDisplay, _current_progress

log = logging.getLogger(__name__)

R = TypeVar('R')

__all__ = ['LocalExecutor']


class LocalExecutor:
    """
    Run functions locally using a thread pool.

    Progress is tracked per-job via ``get_progress()`` inside mapped
    functions, and a live status line is rendered to stderr.
    """

    def __init__(self, name: str, max_workers: int = 1):
        self.name = name
        self.max_workers = max_workers

    def map(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> Iterator[R]:
        kw = kwargs or {}
        args_list = list(zip(*iterables, strict=False))
        n = len(args_list)
        if n == 0:
            return

        log.info('[%s] Running %d jobs with %d workers', self.name, n, self.max_workers)
        display = ProgressDisplay(n)

        def run_one(index: int, args: tuple) -> R:
            progress = display.job_started(index)
            token = _current_progress.set(progress)
            try:
                result = fn(*args, **kw)
                display.job_completed(index)
                return result
            except Exception as e:
                display.job_failed(index, str(e)[:80])
                raise
            finally:
                _current_progress.reset(token)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures: list[Future[R]] = [pool.submit(run_one, i, args) for i, args in enumerate(args_list)]
            try:
                for future in futures:
                    yield future.result()
            finally:
                display.finish()

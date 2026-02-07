"""
Executor protocol and progress reporting infrastructure.

Provides shared components for executor implementations:
- ``Executor`` protocol defining the map interface
- ``ProgressDisplay`` for tracking concurrent job progress
- ``get_progress()`` for mapped functions to report progress

Example::

    from mini.executor import get_progress

    def train(params):
        progress = get_progress()
        if progress:
            progress.set_total(100)
        for epoch in range(100):
            ...
            if progress:
                progress.update(1, message=f"loss={loss:.4f}")
        return result
"""

from __future__ import annotations

import contextvars
import sys
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Protocol, TypeVar

R = TypeVar('R')

__all__ = [
    'Executor',
    'JobProgress',
    'ProgressDisplay',
    'get_progress',
]


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


class ProgressDisplay:
    """
    Aggregate progress from concurrent jobs and render a live status line.

    Each job gets a slot identified by its index.  Jobs report progress via
    ``update()``, and the display refreshes on every update.
    """

    def __init__(self, total_jobs: int):
        self._total_jobs = total_jobs
        self._completed = 0
        self._jobs: dict[int, _JobState] = {}
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._last_line_len = 0

    def job_started(self, job_index: int) -> JobProgress:
        with self._lock:
            state = _JobState()
            self._jobs[job_index] = state
            self._render()
            return JobProgress(job_index, state, self)

    def job_completed(self, job_index: int):
        with self._lock:
            self._completed += 1
            if job_index in self._jobs:
                self._jobs[job_index].status = 'done'
            self._render()

    def job_failed(self, job_index: int, error: str):
        with self._lock:
            self._completed += 1
            if job_index in self._jobs:
                self._jobs[job_index].status = f'FAILED: {error}'
            self._render()

    def _refresh(self):
        """Thread-safe render (acquires the lock)."""
        with self._lock:
            self._render()

    def _render(self):
        """Render the status line.  Caller must hold ``_lock``."""
        elapsed = time.monotonic() - self._start
        parts: list[str] = []
        active = [(idx, s) for idx, s in sorted(self._jobs.items()) if s.status == 'running']
        for idx, s in active[:4]:
            if s.total:
                pct = min(s.step / s.total, 1.0) * 100
                bar_filled = int(pct / 5)
                bar = '#' * bar_filled + '-' * (20 - bar_filled)
                detail = f'[{bar}] {pct:4.0f}%'
            else:
                detail = '...'
            msg = f' {s.message}' if s.message else ''
            parts.append(f'job {idx}: {detail}{msg}')

        status = f'[{elapsed:5.0f}s] {self._completed}/{self._total_jobs} done'
        if parts:
            status += ' | ' + ' | '.join(parts)

        # Overwrite the previous line
        padding = max(0, self._last_line_len - len(status))
        print(f'\r{status}{" " * padding}', end='', flush=True, file=sys.stderr)
        self._last_line_len = len(status)

    def finish(self):
        with self._lock:
            self._render()
            print(file=sys.stderr)  # final newline


class _JobState:
    __slots__ = ('step', 'total', 'message', 'status')

    def __init__(self):
        self.step = 0
        self.total = 0
        self.message = ''
        self.status = 'running'


class JobProgress:
    """Handle for a single job to report its progress.  Thread-safe."""

    def __init__(self, job_index: int, state: _JobState, display: ProgressDisplay):
        self._index = job_index
        self._state = state
        self._display = display

    def set_total(self, total: int):
        self._state.total = total
        self._display._refresh()

    def update(self, n: int = 1, message: str | None = None):
        self._state.step += n
        if message is not None:
            self._state.message = message
        self._display._refresh()

    def set_message(self, message: str):
        self._state.message = message
        self._display._refresh()


# ---------------------------------------------------------------------------
# Context variable — lets mapped functions report progress without needing
# an explicit callback argument.
# ---------------------------------------------------------------------------

_current_progress: contextvars.ContextVar[JobProgress | None] = contextvars.ContextVar(
    'mini_job_progress', default=None
)


def get_progress() -> JobProgress | None:
    """Get the current job's progress handle, or ``None`` if not in an executor."""
    return _current_progress.get()


# ---------------------------------------------------------------------------
# Executor protocol
# ---------------------------------------------------------------------------


class Executor(Protocol):
    """Protocol for running a function over a sweep of inputs."""

    def map(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> Iterator[R]:
        """
        Map *fn* over one or more iterables.

        Like ``concurrent.futures.Executor.map`` and Modal's ``Function.map``:
        the iterables are zipped together and each tuple is unpacked as
        positional arguments.  *kwargs* (if given) are forwarded to every
        call.

        ::

            executor.map(fn, [1, 2, 3])                    # fn(1), fn(2), fn(3)
            executor.map(fn, [1, 2], ['a', 'b'])            # fn(1, 'a'), fn(2, 'b')
            executor.map(fn, [1, 2], kwargs={'k': 'v'})     # fn(1, k='v'), fn(2, k='v')
        """
        ...

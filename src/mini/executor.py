"""
Executor-based experiment runner.

Provides a simple ``map``-style interface for running experiment sweeps,
similar to ``concurrent.futures.ThreadPoolExecutor``.  Functions are passed
in (not decorated), so you can switch between local and remote execution
without changing the experiment code.

Example::

    executor = LocalExecutor("my-experiment", max_workers=2)
    results = list(executor.map(train, [{"seed": 1}, {"seed": 2}]))

    # Switch to Modal for scale
    app = modal.App("my-experiment")
    executor = ModalExecutor(app).with_modal_kwargs(gpu="T4")
    results = list(executor.map(train, [{"seed": i} for i in range(10)]))

Progress reporting
------------------
Mapped functions can optionally report fine-grained progress::

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

``get_progress()`` returns a ``JobProgress`` handle when running inside an
executor, or ``None`` otherwise.  This works for ``LocalExecutor`` (via a
context variable set per-thread).  For ``ModalExecutor``, fine-grained
progress is visible through Modal's built-in log streaming (functions can
simply ``print``); job-level completion is tracked locally.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Iterable, Iterator, Protocol, TypeVar

log = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

__all__ = [
    'Executor',
    'LocalExecutor',
    'ModalExecutor',
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

    def map(self, fn: Callable[..., R], *iterables: Iterable[Any], kwargs: dict[str, Any] | None = None) -> Iterator[R]:
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


# ---------------------------------------------------------------------------
# LocalExecutor
# ---------------------------------------------------------------------------


class LocalExecutor:
    """
    Run functions locally using a thread pool.

    Progress is tracked per-job via ``get_progress()`` inside mapped
    functions, and a live status line is rendered to stderr.
    """

    def __init__(self, name: str, max_workers: int = 1):
        self.name = name
        self.max_workers = max_workers

    def map(self, fn: Callable[..., R], *iterables: Iterable[Any], kwargs: dict[str, Any] | None = None) -> Iterator[R]:
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


# ---------------------------------------------------------------------------
# ModalExecutor
# ---------------------------------------------------------------------------


class ModalExecutor:
    """
    Run functions on Modal.

    Fine-grained progress (epoch-level etc.) is visible through Modal's
    built-in log streaming --- mapped functions can simply ``print()``.
    Job-level completion is tracked locally via a progress display.

    Usage::

        app = modal.App("my-experiment")
        executor = ModalExecutor(app).with_modal_kwargs(gpu="T4", timeout=3600)
        results = list(executor.map(train, configs))
    """

    def __init__(self, app: Any, modal_fn_kwargs: dict[str, Any] | None = None):
        self.app = app
        self.modal_fn_kwargs: dict[str, Any] = modal_fn_kwargs or {}

    def with_modal_kwargs(self, **kwargs: Any) -> ModalExecutor:
        """Return a new executor with additional Modal function kwargs merged in."""
        return ModalExecutor(self.app, {**self.modal_fn_kwargs, **kwargs})

    def map(self, fn: Callable[..., R], *iterables: Iterable[Any], kwargs: dict[str, Any] | None = None) -> Iterator[R]:
        iterables_lists: list[list] = [list(it) for it in iterables]
        n = len(iterables_lists[0]) if iterables_lists else 0
        if n == 0:
            return

        log.info('[ModalExecutor] Running %d jobs on Modal', n)

        # Wrap fn as a Modal function.  The decorator must be applied
        # *before* app.run() starts the app.
        modal_fn = self.app.function(**self.modal_fn_kwargs)(fn)

        display = ProgressDisplay(n)
        # We don't have per-step progress from Modal, but we can track
        # job-level completion as results come back.
        with self.app.run():
            try:
                for i, result in enumerate(modal_fn.map(*iterables_lists, kwargs=kwargs or {})):
                    display.job_completed(i)
                    yield result
            finally:
                display.finish()

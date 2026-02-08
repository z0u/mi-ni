from __future__ import annotations

import contextvars
import sys
import threading
import time

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


def set_progress(progress: JobProgress | None) -> contextvars.Token:
    """Set the current job's progress handle.  Returns a token for resetting."""
    return _current_progress.set(progress)


def reset_progress(token: contextvars.Token):
    """Reset the current job's progress handle using the given token."""
    _current_progress.reset(token)

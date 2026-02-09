from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field

from mini._debounce import Debouncer
from mini._queues import QueueLike
from mini.urns import matches_urn, parse_urn, to_urn

# ---------------------------------------------------------------------------
# Progress message — unified format for all executors
# ---------------------------------------------------------------------------


@dataclass
class ProgressMessage:
    """Structured progress update from a job."""

    run_id: str
    job_id: str
    step: int
    total: int
    message: str = ''

    def __str__(self) -> str:
        return self.to_urn()

    def to_urn(self) -> str:
        """Convert to a URN."""
        return to_urn(
            'mini', 'run', self.run_id, 'progress', self.job_id, str(self.step), str(self.total), self.message
        )

    @classmethod
    def matches(cls, message: str) -> bool:
        return matches_urn(message, 'mini:run:*:progress:*:*:*:*')

    @classmethod
    def from_urn(cls, message: str) -> ProgressMessage:
        """Convert from a URN."""
        parts = parse_urn(message)
        match parts:
            case ('mini', 'run', run_id, 'progress', job_id, step, total, msg):
                return cls(run_id=run_id, job_id=job_id, step=int(step), total=int(total), message=msg)
            case _:
                raise ValueError(f'Invalid progress message format: {message}')


# ---------------------------------------------------------------------------
# Current job context
# ---------------------------------------------------------------------------


@dataclass
class JobContext:
    """Execution context for a job."""

    run_id: str
    job_id: str
    queue: QueueLike[ProgressMessage] | None = None
    emission_interval: float = 0.1
    _emitter: Debouncer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._emitter = Debouncer(self._do_emit, interval=self.emission_interval)

    def _do_emit(self, progress: ProgressMessage) -> None:
        """Actually emit a progress message."""
        if self.queue is not None:
            self.queue.put(progress)
        else:
            print(progress, flush=True)


_job_context: contextvars.ContextVar[JobContext | None] = contextvars.ContextVar('mini_job_context', default=None)


@contextmanager
def progress_context(run_id: str, job_id: str, queue: QueueLike[ProgressMessage] | None, emission_interval: float):
    """Context manager for setting the current job context"""
    ctx = JobContext(
        run_id=run_id,
        job_id=job_id,
        queue=queue,
        emission_interval=emission_interval if emission_interval is not None else 0.1,
    )
    token = _job_context.set(ctx)
    try:
        try:
            yield
        finally:
            ctx._emitter.flush()
    finally:
        _job_context.reset(token)


def emit_progress(step: int, total: int, message: str = ''):
    """
    Emit a progress update for the current job.

    Must be called within a job context. If a progress queue is available, the
    message is queued; otherwise it's printed to stdout.

    Progress emission is debounced per-job with leading and trailing edge semantics:
    - Leading edge: First call emits immediately
    - Trailing edge: Rapid subsequent calls store the latest update and emit after interval
    - Latest arguments: Trailing emission always uses the most recent progress values

    The debounce interval is configured by the executor when setting up the job context.

    Args:
        step: Current step number
        total: Total number of steps
        message: Optional progress message
    """
    ctx = _job_context.get()
    if ctx is None:
        # Silently ignore if not in a job context
        return

    progress = ProgressMessage(run_id=ctx.run_id, job_id=ctx.job_id, step=step, total=total, message=message)
    ctx._emitter(progress)

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field

from mini._debounce import Debouncer
from mini._queues import QueueLike
from mini.urns import matches_urn, parse_urn, to_urn

# ---------------------------------------------------------------------------
# Progress message — unified format for all apparatus
# ---------------------------------------------------------------------------


@dataclass
class ProgressMessage:
    """Structured progress update from a job."""

    run_id: str
    job_id: str
    step: int
    total: int
    message: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.to_urn()

    def to_urn(self) -> str:
        """Convert to a URN."""
        return to_urn(
            "mini", "run", self.run_id, "progress", self.job_id, str(self.step), str(self.total), self.message
        )

    @classmethod
    def matches(cls, message: str) -> bool:
        return matches_urn(message, "mini:run:*:progress:*:*:*:*")

    @classmethod
    def from_urn(cls, message: str) -> ProgressMessage:
        """Convert from a URN."""
        parts = parse_urn(message)
        match parts:
            case ("mini", "run", run_id, "progress", job_id, step, total, msg):
                return cls(run_id=run_id, job_id=job_id, step=int(step), total=int(total), message=msg)
            case _:
                raise ValueError(f"Invalid progress message format: {message}")


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
    metrics: dict[str, float] = field(default_factory=dict)
    _last: tuple[int, int, str] = field(default=(0, 0, ""), init=False, repr=False)
    _emitter: Debouncer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._emitter = Debouncer(self._do_emit, interval=self.emission_interval)

    def _do_emit(self, progress: ProgressMessage) -> None:
        """Actually emit a progress message."""
        if self.queue is not None:
            self.queue.put(progress)
        else:
            print(progress, flush=True)


_job_context: contextvars.ContextVar[JobContext | None] = contextvars.ContextVar("mini_job_context", default=None)


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


def emit_progress(step: int, total: int, message: str = ""):
    """
    Emit a progress update for the current job.

    Must be called within a job context. If a progress queue is available, the
    message is queued; otherwise it's printed to stdout.

    Progress emission is debounced per-job with leading and trailing edge semantics:
    - Leading edge: First call emits immediately
    - Trailing edge: Rapid subsequent calls store the latest update and emit after interval
    - Latest arguments: Trailing emission always uses the most recent progress values

    The debounce interval is configured by the apparatus when setting up the job context.

    Args:
        step: Current step number
        total: Total number of steps
        message: Optional progress message
    """
    ctx = _job_context.get()
    if ctx is None:
        # Silently ignore if not in a job context
        return

    ctx._last = (step, total, message)
    progress = ProgressMessage(
        run_id=ctx.run_id, job_id=ctx.job_id, step=step, total=total, message=message, metrics=dict(ctx.metrics)
    )
    ctx._emitter(progress)


def emit_metrics(**scalars: float) -> None:
    """
    Report the latest scalar metrics for the current job (e.g. ``loss``, ``lr``).

    Like ``emit_progress``, this is a no-op outside a job context. Metrics are
    merged (last-writer-wins) and travel with progress updates, so a watching
    agent or human can see a run's numbers — and step in if they go awry —
    without waiting for it to finish.
    """
    ctx = _job_context.get()
    if ctx is None:
        return

    ctx.metrics.update(scalars)
    step, total, message = ctx._last
    ctx._emitter(
        ProgressMessage(
            run_id=ctx.run_id, job_id=ctx.job_id, step=step, total=total, message=message, metrics=dict(ctx.metrics)
        )
    )

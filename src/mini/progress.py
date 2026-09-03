from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field

from mini._debounce import BackgroundEmitter
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
    # Which way each metric is *supposed* to move ("up" / "down"), as declared by
    # :func:`expect_metrics`. Travels with the numbers because only the job knows
    # it — see there for why it isn't inferred downstream.
    goals: dict[str, str] = field(default_factory=dict)

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
    # Called synchronously, on the emitting thread, with each ``(step, total)`` an
    # ``emit_progress`` reports. The watchdog rides here rather than on the sink so
    # it measures the *task's* progress: delivery happens on a background thread, so
    # a slow control plane would otherwise read as a wedged worker.
    on_progress: Callable[[int, int], None] | None = None
    # Opens a declared step-free span (see :func:`blocking_phase`). The worker
    # binds it to the watchdog's own ``phase`` plus a record stamp, so both the
    # abort threshold and the client-side staleness badge learn about the span.
    on_phase: Callable[[str, float], AbstractContextManager[None]] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    goals: dict[str, str] = field(default_factory=dict)
    _last: tuple[int, int, str] = field(default=(0, 0, ""), init=False, repr=False)
    _emitter: BackgroundEmitter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._emitter = BackgroundEmitter(
            self._do_emit, interval=self.emission_interval, name=f"mini-emit-{self.job_id}"
        )

    def _do_emit(self, progress: ProgressMessage) -> None:
        """Actually emit a progress message (on the emitter's thread)."""
        if self.queue is not None:
            self.queue.put(progress)
        else:
            print(progress, flush=True)


_job_context: contextvars.ContextVar[JobContext | None] = contextvars.ContextVar("mini_job_context", default=None)


@contextmanager
def progress_context(
    run_id: str,
    job_id: str,
    queue: QueueLike[ProgressMessage] | None,
    emission_interval: float,
    on_progress: Callable[[int, int], None] | None = None,
    on_phase: Callable[[str, float], AbstractContextManager[None]] | None = None,
):
    """Context manager for setting the current job context"""
    ctx = JobContext(
        run_id=run_id,
        job_id=job_id,
        queue=queue,
        emission_interval=emission_interval if emission_interval is not None else 0.1,
        on_progress=on_progress,
        on_phase=on_phase,
    )
    token = _job_context.set(ctx)
    try:
        try:
            yield
        finally:
            ctx._emitter.close()  # let the last update land before the record settles
    finally:
        _job_context.reset(token)


def emit_progress(step: int, total: int, message: str = ""):
    """
    Emit a progress update for the current job.

    Must be called within a job context. If a progress queue is available, the
    message is queued; otherwise it's printed to stdout.

    **This call never blocks on the sink.** Delivery happens on a per-job daemon
    thread holding one latest-wins slot: a call while an emission is in flight
    replaces the pending update rather than queueing behind it, so a task emitting
    every step runs at its own speed and simply reports coarser progress when the
    sink is slow (see :class:`~mini._debounce.BackgroundEmitter`). *interval*, set
    by the apparatus, is a floor on the spacing, not a promise of one.

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
    if ctx.on_progress is not None:
        ctx.on_progress(step, total)  # synchronous and local — the watchdog's advance signal
    ctx._emitter(_message(ctx, step, total, message))


@contextmanager
def blocking_phase(label: str, timeout_s: float) -> Iterator[None]:
    """Declare a span that legitimately makes no step progress.

    The worker's progress watchdog aborts a task whose ``(step, total)`` stops advancing, which is what turns a wedged worker into a fast retryable failure instead of a burnt role timeout. But some spans have no steps to report and are perfectly healthy: uploading a checkpoint after the last step, pulling a dataset before the first, a long eval between epochs. Name one, and the watchdog measures it against *timeout_s* instead of the tight step threshold::

        with blocking_phase("upload checkpoint", timeout_s=600):
            art = put(workdir / "model", name="model")

    The budget is a bound rather than a pass: a span that hangs is still caught, just at *timeout_s*. Size it for the slowest healthy case — aborting a healthy task throws away everything it has done, while catching a wedge late only costs idle time.

    ``mini.store``'s ``put``, ``get`` and ``get_many`` already declare their own, sized from the payload, so a step that only moves artifacts needs nothing here. Phases nest, and only ever widen the threshold.

    A no-op outside a job context, and on a worker with no watchdog armed.
    """
    ctx = _job_context.get()
    with nullcontext() if ctx is None or ctx.on_phase is None else ctx.on_phase(label, timeout_s):
        yield


def emit_metrics(**scalars: float) -> None:
    """
    Report the latest scalar metrics for the current job (e.g. ``loss``, ``lr``).

    Like ``emit_progress``, this is a no-op outside a job context. Metrics are merged (last-writer-wins) and travel with progress updates, so a watching agent or human can see a run's numbers — and step in if they go awry — without waiting for it to finish.

    Pair it with :func:`expect_metrics` for anything whose direction isn't obvious from its name, and the tools will flag a metric heading the wrong way.
    """
    ctx = _job_context.get()
    if ctx is None:
        return

    ctx.metrics.update(scalars)
    ctx._emitter(_message(ctx, *ctx._last))


def expect_metrics(**directions: str) -> None:
    """
    Declare which way each metric is *supposed* to move: ``"up"`` or ``"down"``.

    ``expect_metrics(loss="down", accuracy="up")``. With this, ``status`` can flag a metric drifting the wrong way for several windows — a sliding accuracy reads exactly as loudly as a climbing loss.

    The declaration lives here, in the job, because this is the only place that knows. A name is a poor proxy: ``loss_scale`` is a mixed-precision scale factor that climbs quite happily, and a domain-specific score gives nothing away at all. So the worker guesses only for a few unambiguous names (``loss``, ``val_loss``, ``nll``, ``perplexity``, …) and stays quiet about the rest — silence rather than a wrong guess. Declaring is how you opt a metric in.

    A no-op outside a job context, and safe to call repeatedly (later calls win). Call it before the loop; it rides along on every subsequent update.
    """
    if bad := {k: v for k, v in directions.items() if v not in ("up", "down")}:
        raise ValueError(f"metric directions must be 'up' or 'down', got {bad}")
    if (ctx := _job_context.get()) is not None:
        ctx.goals.update(directions)


def _message(ctx: JobContext, step: int, total: int, message: str) -> ProgressMessage:
    return ProgressMessage(
        run_id=ctx.run_id,
        job_id=ctx.job_id,
        step=step,
        total=total,
        message=message,
        metrics=dict(ctx.metrics),
        goals=dict(ctx.goals),
    )

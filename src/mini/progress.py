from __future__ import annotations

import contextvars
from dataclasses import dataclass

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
# Context variables — track current job and optional progress queue
# ---------------------------------------------------------------------------

_current_job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar('mini_job_id', default=None)
_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar('mini_run_id', default=None)
_progress_queue: contextvars.ContextVar[QueueLike[ProgressMessage] | None] = contextvars.ContextVar('mini_progress_queue', default=None)


def set_job_context(
    run_id: str, job_id: str, queue: QueueLike[ProgressMessage] | None = None
) -> tuple[contextvars.Token, contextvars.Token, contextvars.Token | None]:
    """Set the current job context. Returns tokens for resetting."""
    token1 = _current_run_id.set(run_id)
    token2 = _current_job_id.set(job_id)
    token3 = _progress_queue.set(queue) if queue is not None else None
    return token1, token2, token3


def reset_job_context(token1: contextvars.Token, token2: contextvars.Token, token3: contextvars.Token | None = None):
    """Reset the current job context using the given tokens."""
    _current_run_id.reset(token1)
    _current_job_id.reset(token2)
    if token3 is not None:
        _progress_queue.reset(token3)


def emit_progress(step: int, total: int, message: str = ''):
    """
    Emit a progress update for the current job.

    Must be called within a job context where run_id and job_id are set.
    If a progress queue is available (via set_progress_queue), the message
    is queued; otherwise it's printed to stdout.
    """
    run_id = _current_run_id.get()
    job_id = _current_job_id.get()

    if run_id is None or job_id is None:
        # Silently ignore if not in a job context (e.g., running outside an executor)
        return

    progress = ProgressMessage(run_id=run_id, job_id=job_id, step=step, total=total, message=message)

    # Try to queue the message if a queue is available
    queue = _progress_queue.get()
    if queue is not None:
        queue.put(progress)
    else:
        # Fall back to printing
        print(progress, flush=True)


def set_progress_queue(queue: QueueLike[ProgressMessage]) -> contextvars.Token:
    """Set a queue for collecting progress messages. Returns a token for resetting."""
    return _progress_queue.set(queue)


def reset_progress_queue(token: contextvars.Token):
    """Reset the progress queue using the given token."""
    _progress_queue.reset(token)

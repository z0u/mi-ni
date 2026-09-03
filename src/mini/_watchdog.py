"""
Worker-side progress watchdog: abort a wedged task instead of letting it burn.

A wedged worker (hung device call, deadlocked thread) can sit on its GPU allocations while making no progress — and backend liveness probes still see a healthy container, so the wedge silently burns its whole role ``timeout`` (seen in ex-2.1.4: 45 minutes at 0.3 % GPU utilization). Frozen *step* progress is the one honest signal: heartbeats can keep beating from a side thread, but a stalled training loop stops advancing ``step``.

The watchdog turns that silent stall into a fast, retryable failure: if the ``(step, total)`` pair hasn't advanced within ``timeout_s``, it settles the task's record FAILED (with a stack dump of every thread — the closest thing a wedge has to a traceback) and hard-exits the process, releasing the GPU. ``os._exit`` because a thread stuck in a native call never returns to the interpreter, so raising an exception at it could go unheard forever.

Heartbeats and metrics-only emissions deliberately do **not** feed the watchdog — liveness without progress is precisely the wedge signature.

Some spans have no steps to report and are healthy anyway: a checkpoint upload after the last step, a dataset pull before the first. :meth:`Watchdog.phase` lets the task declare those and name a budget for them, so they are measured against their own threshold rather than the tight step one.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import count
from typing import Any, Callable

__all__ = ["Watchdog", "WatchdogStall"]


class WatchdogStall(RuntimeError):
    """A task's step progress stalled past the watchdog timeout; the worker aborted itself."""


class Watchdog:
    """Abort the process when step progress stalls for *timeout_s* seconds.

    Use as a context manager around the task call; feed it via :meth:`poke` from the progress sink.

    Until the first poke, the effective threshold is *grace_s* (defaulting to *timeout_s*): one-off setup — tokenizing a dataset, compiling a model — happens before the first ``emit_progress``, and without a separate grace it would force the whole watchdog loose (a 10-minute prep phase would demand a 10-minute timeout even when training steps take seconds). After the first advance the tight *timeout_s* takes over, so size it past the longest legitimate gap *between* step advances only. The grace ends at the first emission — a task that emits step 0 up front and then preps for minutes should hold its first emission until real step cadence begins.

    Spans in the *middle* of a task that legitimately make no step progress get :meth:`phase`, which is the same idea as the grace with a caller-named budget.

    On stall, *on_stall* is called with a diagnosis (summary + all-thread stack dump) — it should persist the failure wherever the task records state — and then the process exits, no matter what *on_stall* did. Exiting on the watchdog's own thread is the point: the wedged main thread can't be relied on to run anything.
    """

    def __init__(
        self,
        timeout_s: float,
        on_stall: Callable[[str], None],
        grace_s: float | None = None,
        _exit: Callable[[int], Any] = os._exit,
    ):
        self.timeout_s = timeout_s
        self.grace_s = grace_s if grace_s is not None else timeout_s
        self._on_stall = on_stall
        self._exit = _exit
        self._last: tuple[int, int] | None = None
        self._advanced_at = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mini-watchdog", daemon=True)
        # Blocking phases currently open, by entry token. Keyed rather than a plain
        # list so nested phases pop the one they pushed, not an equal-valued sibling.
        self._phases: dict[int, tuple[float, str]] = {}
        self._phase_tokens = count()
        self._lock = threading.Lock()  # _phases is read from the watchdog thread

    def poke(self, step: int, total: int) -> None:
        """Report the task's current position; only a *change* resets the clock."""
        if (step, total) != self._last:
            self._last = (step, total)
            self._advanced_at = time.monotonic()

    @contextmanager
    def phase(self, label: str, timeout_s: float) -> Iterator[None]:
        """Allow *timeout_s* without step progress for a declared blocking span.

        Setup gets ``grace_s``; this is the same allowance for any span the task names as legitimately step-free — uploading a checkpoint, pulling a dataset, a long eval between epochs. The clock resets on both entry and exit, so the span gets its whole budget and the loop that follows gets its whole timeout.

        A budget only ever *widens* the threshold: it is a bound, not a pass. A span that truly hangs is still caught, at *timeout_s* rather than at the tight step timeout. Size it for the slowest healthy case — aborting a healthy task throws away everything it has done, while catching a wedge late only costs idle time.

        Nests, and is safe to call from any thread.
        """
        token = next(self._phase_tokens)
        with self._lock:
            self._phases[token] = (timeout_s, label)
            self._advanced_at = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                self._phases.pop(token, None)
                self._advanced_at = time.monotonic()

    def _widest_phase(self) -> tuple[float, str] | None:
        with self._lock:
            phases = list(self._phases.values())
        return max(phases, key=lambda p: p[0]) if phases else None

    def __enter__(self) -> Watchdog:
        self._advanced_at = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()

    def _standing_threshold(self) -> float:
        """What governs absent any declared phase: the grace until the first poke."""
        return self.timeout_s if self._last is not None else self.grace_s

    def _threshold(self) -> float:
        phase = self._widest_phase()
        standing = self._standing_threshold()
        return max(standing, phase[0]) if phase is not None else standing

    def _run(self) -> None:
        poll = min(self.timeout_s / 4, self.grace_s / 4, 15.0)
        while not self._stop.wait(poll):
            stalled_s = time.monotonic() - self._advanced_at
            if stalled_s < self._threshold() or self._stop.is_set():
                continue
            try:
                self._on_stall(self._diagnosis(stalled_s))
            finally:
                self._exit(70)  # EX_SOFTWARE; unconditional — the main thread is presumed wedged
            return

    def _diagnosis(self, stalled_s: float) -> str:
        """All-thread stack dump ending in the stall summary (traceback-shaped: the last line is the failure, so record summaries read naturally)."""
        names = {t.ident: t.name for t in threading.enumerate()}
        parts = ["Watchdog stall — stacks of all threads at abort:\n"]
        for ident, frame in sys._current_frames().items():
            label = names.get(ident, "?")
            suffix = " (this watchdog)" if ident == threading.get_ident() else ""
            parts.append(f"--- thread {label}{suffix} ---\n{''.join(traceback.format_stack(frame))}")
        if self._last:
            at, limit = f" at step {self._last[0]}/{self._last[1]}", f"watchdog {self.timeout_s:g}s"
        else:
            at, limit = " before any progress emission", f"startup grace {self.grace_s:g}s"
        # Name the threshold that actually expired, so the diagnosis says which
        # budget to look at — the tight step one, or a phase that outlasted it.
        if (phase := self._widest_phase()) is not None and phase[0] > self._standing_threshold():
            limit = f"blocking phase {phase[1]!r} {phase[0]:g}s"
        parts.append(
            f"WatchdogStall: no step progress in {stalled_s:.0f}s ({limit}){at}"
            " — worker aborted, releasing its resources; retry when ready"
        )
        return "\n".join(parts)

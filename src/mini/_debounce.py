from __future__ import annotations

from collections.abc import Callable
import logging
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Debouncer — generic, reusable debounce with leading + trailing edge
# ---------------------------------------------------------------------------


class Debouncer:
    """
    Debounce calls to a function with leading and trailing edge semantics.

    - **Leading edge:** first call (or first call after the interval elapses) fires immediately.
    - **Trailing edge:** rapid subsequent calls store only the latest arguments and emit once after the interval.

    Thread-safe. The trailing-edge timer is daemonic, so it won't prevent program exit — call :meth:`flush` to guarantee delivery.
    """

    def __init__(self, fn: Callable[..., Any], interval: float = 0.1) -> None:
        self._fn = fn
        self._interval = interval
        self._lock = threading.Lock()
        self._last_emission = 0.0
        self._pending: tuple[tuple, dict] | None = None
        self._timer: threading.Timer | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Invoke the wrapped function, debouncing as needed."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_emission >= self._interval:
                self._fn(*args, **kwargs)
                self._last_emission = now
                if self._timer:
                    self._timer.cancel()
                    self._timer = None
            else:
                self._pending = (args, kwargs)
                if self._timer:
                    self._timer.cancel()
                delay = self._interval - (now - self._last_emission)

                def _emit_pending() -> None:
                    with self._lock:
                        if self._pending:
                            a, kw = self._pending
                            self._fn(*a, **kw)
                            self._last_emission = time.monotonic()
                            self._pending = None
                            self._timer = None

                timer = threading.Timer(delay, _emit_pending)
                timer.daemon = True
                timer.start()
                self._timer = timer

    def flush(self) -> None:
        """Flush any pending call immediately."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            if self._pending:
                args, kwargs = self._pending
                self._fn(*args, **kwargs)
                self._last_emission = time.monotonic()
                self._pending = None


# ---------------------------------------------------------------------------
# BackgroundEmitter — hand the latest value to a slow sink, off the caller's thread
# ---------------------------------------------------------------------------


class BackgroundEmitter:
    """Deliver the most recent call's arguments to a slow *fn*, without blocking.

    A debouncer runs its leading edge on the calling thread, which is fine when the sink is local and quick. When the sink is a network write — a Modal Queue put, a control-plane record merge — and its latency grows past the debounce interval, *every* call fires the leading edge and the caller degrades to one blocking round-trip per call. A training loop emitting progress each step then runs at the speed of the network rather than the GPU (diagnosed in ex-2.1.5: containers far from the control plane ran 15–30× slow, in order of distance).

    So: one slot, latest-wins, drained by a daemon thread. The caller stores its arguments and returns immediately; the thread delivers whatever is in the slot when it gets there and skips whatever was superseded meanwhile. Delivery rate self-limits to the sink's own speed — a slow sink just means coarser progress, which is exactly the right thing to trade away — and *interval* sets a floor on the spacing so a fast sink isn't hammered.

    Sink failures are logged and dropped: progress is diagnostic, and the caller is no longer in a position to handle them anyway.
    """

    def __init__(self, fn: Callable[..., Any], interval: float = 0.1, name: str = "mini-emit") -> None:
        self._fn = fn
        self._interval = interval
        self._name = name
        self._cv = threading.Condition()
        self._pending: tuple[tuple, dict] | None = None
        self._busy = False
        self._closed = False
        self._urgent = False  # someone is waiting on a flush — skip the pacing hold
        self._thread: threading.Thread | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Queue *args* for delivery, replacing anything not yet sent. Never blocks."""
        with self._cv:
            if self._closed:
                return
            self._pending = (args, kwargs)
            if self._thread is None:  # started on first use: a job that never emits pays nothing
                self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
                self._thread.start()
            self._cv.notify_all()

    def _run(self) -> None:
        sent_at = 0.0
        while True:
            with self._cv:
                while self._pending is None and not self._closed:
                    self._cv.wait()
                if self._pending is None:
                    return  # closed and drained
                # Hold the interval since the last delivery — the floor on spacing.
                # A waiting flush (or a close) skips it: the point of the pause is to
                # spare a fast sink, not to make a job wait on its final update.
                while (
                    not self._urgent and not self._closed and (left := self._interval + sent_at - time.monotonic()) > 0
                ):
                    self._cv.wait(left)
                (args, kwargs), self._pending = self._pending, None
                self._busy = True
            try:
                self._fn(*args, **kwargs)
            except Exception:
                log.debug("progress emission failed; dropping it", exc_info=True)
            finally:
                sent_at = time.monotonic()
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait (up to *timeout*) for the slot to drain. True if it did.

        The end-of-job barrier: a task's last progress update should land before its record settles, and an in-flight write should finish before the process exits.
        """
        deadline = time.monotonic() + timeout
        with self._cv:
            self._urgent = True
            self._cv.notify_all()
            try:
                while self._thread is not None and (self._pending is not None or self._busy):
                    if not self._cv.wait(max(0.0, deadline - time.monotonic())):
                        return False
                return True
            finally:
                self._urgent = False

    def close(self, timeout: float = 30.0) -> None:
        """Flush, then stop the thread (it's a daemon, so this is tidiness, not safety)."""
        self.flush(timeout)
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

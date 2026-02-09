from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Any

# ---------------------------------------------------------------------------
# Debouncer — generic, reusable debounce with leading + trailing edge
# ---------------------------------------------------------------------------


class Debouncer:
    """
    Debounce calls to a function with leading and trailing edge semantics.

    - **Leading edge:** first call (or first call after the interval elapses)
      fires immediately.
    - **Trailing edge:** rapid subsequent calls store only the latest arguments
      and emit once after the interval.

    Thread-safe. The trailing-edge timer is daemonic, so it won't prevent
    program exit — call :meth:`flush` to guarantee delivery.
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

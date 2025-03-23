import logging
import sys
import time
from dataclasses import dataclass
from typing import Literal, TypeAlias, override

TRACE = 5


class ConciseFormatter(logging.Formatter):
    """
    Custom formatter that includes elapsed time since program start.

    Example usage:
        ```python
        import logging
        handler = logging.StreamHandler(stream)
        handler.setFormatter(ConciseFormatter())
        logging.basicConfig(level=logging.WARNING, handlers=[handler])
        ```

    Example output for 'package.module':
        ```
        W    5.1 p.m:  message
        ```
    """

    start_time = time.monotonic()

    @override
    def format(self, record):
        elapsed = time.monotonic() - self.start_time
        abbreviated_log_level = record.levelname[0]
        abbreviated_module_name = '.'.join(p[:2] for p in record.name.split('.'))

        # Format the message
        prefix = f'{abbreviated_log_level} {elapsed:.1f} {abbreviated_module_name}:'
        return f'{prefix:15s}{record.getMessage()}'


NamedFd: TypeAlias = Literal['stdout', 'stderr']


@dataclass
class SimpleLoggingConfig:
    """Build a concise logging configuration with a fluent interface."""

    def __init__(self):
        # These need to be serializable
        self._base_level: int = logging.WARNING
        self._stream: NamedFd = 'stdout'
        self._critical: list[str] = []
        self._error: list[str] = []
        self._warning: list[str] = []
        self._info: list[str] = []
        self._debug: list[str] = []
        self._trace: list[str] = []

    def __call__(self):
        """
        Apply the logging configuration.

        Alias for `apply()`.
        """
        return self.apply()

    def base_level(self, level: int):
        """Set the root logging level."""
        self._base_level = level
        return self

    def to_stream(self, stream: NamedFd):
        """Set the output stream."""
        self._stream = stream
        return self

    def critical(self, *names):
        """Add loggers at CRITICAL level."""
        self._critical.extend(names)
        return self

    def error(self, *names):
        """Add loggers at ERROR level."""
        self._error.extend(names)
        return self

    def warning(self, *names):
        """Add loggers at WARNING level."""
        self._warning.extend(names)
        return self

    def info(self, *names):
        """Add loggers at INFO level."""
        self._info.extend(names)
        return self

    def debug(self, *names):
        """Add loggers at DEBUG level."""
        self._debug.extend(names)
        return self

    def trace(self, *names):
        """Add loggers at TRACE level."""
        self._trace.extend(names)
        return self

    def apply(self):
        """Apply the logging configuration."""
        logging.addLevelName(TRACE, 'TRACE')

        handler = logging.StreamHandler(sys.stdout if self._stream == 'stdout' else sys.stderr)
        handler.setFormatter(ConciseFormatter())
        logging.basicConfig(level=self._base_level, handlers=[handler])

        for name in self._critical:
            logging.getLogger(name).setLevel(logging.CRITICAL)
        for name in self._error:
            logging.getLogger(name).setLevel(logging.ERROR)
        for name in self._warning:
            logging.getLogger(name).setLevel(logging.WARNING)
        for name in self._info:
            logging.getLogger(name).setLevel(logging.INFO)
        for name in self._debug:
            logging.getLogger(name).setLevel(logging.DEBUG)
        for name in self._trace:
            logging.getLogger(name).setLevel(TRACE)

        return self  # For method chaining if needed

import logging
import sys
import time
from typing import override


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

    Example output:
        ```
        W    5.1 p.m:  message
        ```
    """

    start_time = time.monotonic()

    @override
    def format(self, record):
        elapsed = time.monotonic() - self.start_time
        abbreviated_log_level = record.levelname[0]
        abbreviated_module_name = '.'.join(p[0] for p in record.name.split('.'))

        # Format the message
        prefix = f'{abbreviated_log_level} {elapsed:.1f} {abbreviated_module_name}:'
        return f'{prefix:15s}{record.getMessage()}'


def concise_logging(level=logging.WARNING, stream=sys.stderr):
    """
    Create a concise logging handler that outputs to the given stream.

    Example usage:
        ```python
        from utils.logging import concise_logging
        concise_logging()
        ```
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ConciseFormatter())
    logging.basicConfig(level=level, handlers=[handler])

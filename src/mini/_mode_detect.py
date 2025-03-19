from asyncio import iscoroutinefunction
from inspect import isasyncgenfunction
import logging

log = logging.getLogger(__name__)


def detect_mode(callback):
    # Auto-detect context manangers
    if hasattr(callback, '__aenter__') and hasattr(callback, '__aexit__'):
        # Context manager instance
        log.debug('Auto-detected context manager instance')
        return 'cm'
    elif hasattr(callback, '__wrapped__') and isasyncgenfunction(callback.__wrapped__):
        # Function decorated with @asynccontextmanager
        log.debug('Auto-detected context manager factory (@asynccontextmanager)')
        return 'cm_factory'

    # Detect bare callback vs factory by checking if it's async
    if not iscoroutinefunction(callback):
        # It's a synchronous function - must be a factory!
        log.debug('Auto-detected factory function (not a coroutine function)')
        return 'factory'
    else:
        # It's an async function - regular callback
        log.debug('Auto-detected regular async callback')
        return 'callback'

from contextlib import asynccontextmanager

from mini._mode_detect import detect_mode


async def async_callback(x, y):
    pass


class AsyncContextManager:
    async def __aenter__(self):
        return async_callback

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        pass


@asynccontextmanager
async def async_cm_factory():
    yield async_callback


def sync_factory():
    return async_callback


def test_context_manager_instance():
    cm = AsyncContextManager()
    assert detect_mode(cm) == 'cm'


def test_context_manager_factory():
    assert detect_mode(async_cm_factory) == 'cm_factory'


def test_sync_factory():
    assert detect_mode(sync_factory) == 'factory'


def test_async_callback():
    assert detect_mode(async_callback) == 'callback'

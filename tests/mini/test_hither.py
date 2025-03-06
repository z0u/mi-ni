from contextlib import asynccontextmanager
from typing import List
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from mini.hither import (
    _run_hither,
    _run_hither_batch,
    _run_hither_batch_cm,
    _run_hither_cm,
    run_hither,
    run_hither_batch,
)
from mini.types import Params, SyncHandler


# Test fixtures
async def stub_callback(x: int, y: str) -> None:
    """A test callback."""  # noqa: D401
    pass


def stub_cb_factory():
    return stub_callback


async def stub_batch_callback(items: List[int]) -> None:
    pass


def stub_batch_cb_factory():
    return stub_batch_callback


@asynccontextmanager
async def decorated_stub_cb_cm():
    yield stub_callback


@asynccontextmanager
async def decorated_stub_batch_cb_cm():
    yield stub_batch_callback


class StubCallbackContextManager:
    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return stub_callback

    async def __aexit__(self, *args):
        self.exit_count += 1


class StubBatchCallbackContextManager:
    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0
        # self.callback_spy = typed_async_mock(stub_batch_callback)

    async def __aenter__(self):
        self.enter_count += 1
        return stub_batch_callback

    async def __aexit__(self, *args):
        self.exit_count += 1


@pytest.fixture
def mock_send_batch_to[T]():
    with patch('mini.hither.send_batch_to') as mock:
        mock_context = AsyncMock()
        mock_send_batch: SyncHandler[list[T]] | Mock = Mock()
        mock_context.__aenter__.return_value = mock_send_batch
        mock.return_value = mock_context
        yield mock, mock_context, mock_send_batch


async def test_run_hither_basic(mock_send_batch_to):
    _, _, mock_send_batch = mock_send_batch_to

    async with _run_hither(stub_callback) as callback:
        callback(1, 'test')

    # Verify the call was made correctly
    mock_send_batch.assert_called_once()
    args = mock_send_batch.call_args[0][0]
    assert len(args) == 1
    assert args[0].args == (1, 'test')
    assert args[0].kwargs == {}


async def test_run_hither_batch(mock_send_batch_to):
    _, _, mock_send_batch = mock_send_batch_to

    async with _run_hither_batch(stub_batch_callback) as callback:
        callback(1)
        callback(2)
        callback(3)

    mock_send_batch.assert_any_call([1])
    mock_send_batch.assert_any_call([2])
    mock_send_batch.assert_any_call([3])


async def test_run_hither_cm(mock_send_batch_to):
    """Test that context managers are correctly handled."""
    _, _, mock_send_batch = mock_send_batch_to

    # Create a real context manager but with spies for its methods
    cm = StubCallbackContextManager()

    async with _run_hither_cm(cm) as callback:
        # Verify the spies were called correctly
        assert cm.enter_count == 1, 'Context manager __aenter__ was not called'
        assert cm.exit_count == 0, 'Context manager __aexit__ should not be called yet'

        callback(1, 'test')

        # Verify the send_batch was called
        mock_send_batch.assert_called_with([Params(args=(1, 'test'), kwargs={})])

    # After context exits, __aexit__ should be called
    assert cm.exit_count == 1, 'Context manager __aexit__ should be called after exiting the context'


async def test_run_hither_batch_cm(mock_send_batch_to):
    """Test that batch context managers are correctly handled."""
    _, _, mock_send_batch = mock_send_batch_to

    # Create a real context manager but with spies for its methods
    cm = StubBatchCallbackContextManager()

    async with _run_hither_batch_cm(cm) as callback:
        # Verify the spies were called correctly
        assert cm.enter_count == 1, 'Context manager __aenter__ was not called'
        assert cm.exit_count == 0, 'Context manager __aexit__ should not be called yet'
        callback(1)
        callback(2)
        callback(3)

    mock_send_batch.assert_any_call([1])
    mock_send_batch.assert_any_call([2])
    mock_send_batch.assert_any_call([3])

    # After context exits, __aexit__ should be called
    assert cm.exit_count == 1, 'Context manager __aexit__ should be called after exiting the context'


async def test_batched_callback_execution(mock_send_batch_to):
    """Test that _run_hither correctly processes each call in the batch."""
    callback_mock = AsyncMock()

    mock, _, _ = mock_send_batch_to  # Get the first mock object correctly

    async with _run_hither(callback_mock) as _:
        # We don't call the send function directly here, because we want to test that the internal function is called correctly.
        pass

    # Get the batched_callback that was passed to send_batch_to - fix the access pattern
    batched_callback = mock.call_args[0][0]

    # Create a batch of calls
    calls = [Params((1, 'a'), {}), Params((2, 'b'), {'extra': True}), Params((3, 'c'), {})]

    # Call the batched_callback with this batch
    await batched_callback(calls)

    # Verify callback was called for each item in the batch
    assert callback_mock.call_count == 3
    callback_mock.assert_has_calls([call(1, 'a'), call(2, 'b', extra=True), call(3, 'c')])


async def test_run_hither_callback(mock_send_batch_to):
    """run_hither(callback)"""
    with patch('mini.hither._run_hither') as mock_run:
        mock_run.return_value.__aenter__.return_value = 'test'

        # run_hither(callback) should call _run_hither(callback)
        async with run_hither(stub_callback) as _:
            pass

        mock_run.assert_called_once_with(stub_callback)


async def test_run_hither_factory(mock_send_batch_to):
    """run_hither(() -> callback) ≍ run_hither(callback)"""
    with patch('mini.hither._run_hither') as mock_run:
        mock_run.return_value.__aenter__.return_value = 'test'

        async with run_hither(stub_cb_factory)() as _:
            pass

        mock_run.assert_called_once_with(stub_callback)


async def test_run_hither_context_manager(mock_send_batch_to):
    """run_hither(with -> callback) ≍ run_hither(callback)"""
    with patch('mini.hither._run_hither') as _run_hither:
        _run_hither.return_value.__aenter__.return_value = 'test'
        cm = StubCallbackContextManager()

        async with run_hither(cm) as _:
            pass

        _run_hither.assert_called_once_with(stub_callback)


async def test_run_hither_decorated_context_manager(mock_send_batch_to):
    """run_hither(with() -> callback)() ≍ run_hither(callback)"""
    with patch('mini.hither._run_hither') as _run_hither:
        _run_hither.return_value.__aenter__.return_value = 'test'

        # Unlike run_hither(AsyncContextManager), run_hither(@asynccontextmanager) returns a function that makes the context manager, just like @asynccontextmanager itself.
        async with run_hither(decorated_stub_cb_cm)() as _:
            pass

        _run_hither.assert_called_once_with(stub_callback)


async def test_run_hither_batch_callback(mock_send_batch_to):
    """run_hither_batch(callback)"""
    with patch('mini.hither._run_hither_batch') as _run_hither_batch:
        _run_hither_batch.return_value.__aenter__.return_value = 'test'

        async with run_hither_batch(stub_batch_callback) as _:
            pass

        _run_hither_batch.assert_called_once_with(stub_batch_callback)


async def test_run_hither_batch_callback_factory(mock_send_batch_to):
    """run_hither_batch(() -> callback)() ≍ run_hither_batch(callback)"""
    with patch('mini.hither._run_hither_batch') as _run_hither_batch:
        _run_hither_batch.return_value.__aenter__.return_value = 'test'

        async with run_hither_batch(stub_batch_cb_factory)() as _:
            pass

        _run_hither_batch.assert_called_once_with(stub_batch_callback)


async def test_run_hither_batch_context_manager(mock_send_batch_to):
    """run_hither_batch(with -> callback)() ≍ run_hither_batch(callback)"""
    with patch('mini.hither._run_hither_batch') as _run_hither_batch:
        _run_hither_batch.return_value.__aenter__.return_value = 'test'
        cm = StubBatchCallbackContextManager()

        async with run_hither_batch(cm) as _:
            pass

        _run_hither_batch.assert_called_once_with(stub_batch_callback)


async def test_run_hither_batch_decorated_context_manager(mock_send_batch_to):
    """run_hither_batch(with() -> callback)() ≍ run_hither_batch(callback)"""
    with patch('mini.hither._run_hither_batch') as _run_hither_batch:
        _run_hither_batch.return_value.__aenter__.return_value = 'test'

        async with run_hither_batch(decorated_stub_batch_cb_cm)() as _:
            pass

        _run_hither_batch.assert_called_once_with(stub_batch_callback)

from contextlib import asynccontextmanager
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from mini.hither import (
    _run_hither,
    _run_hither_batch,
    _run_hither_batch_cm,
    _run_hither_batch_cm_factory,
    _run_hither_batch_factory,
    _run_hither_cm,
    _run_hither_cm_factory,
    _run_hither_factory,
    run_hither,
)
from mini.types import Params, SyncHandler
from tests.test_utils.typed_mocks import typed_async_mock


# Test fixtures
async def stub_callback(x: int, y: str) -> None:
    """A test callback."""  # noqa: D401
    pass


def stub_cb_factory():
    async def inner_callback(x: int, y: str) -> None:
        pass

    return inner_callback


async def stub_batch_callback(items: List[int]) -> None:
    pass


def stub_batch_cb_factory():
    async def inner_batch_callback(items: List[int]) -> None:
        pass

    return inner_batch_callback


@asynccontextmanager
async def stub_cb_cm():
    async def inner_cm(x: int, y: str) -> None:
        pass

    yield inner_cm


@asynccontextmanager
async def stub_batch_cb_cm():
    async def inner_batch_cm(items: List[int]) -> None:
        pass

    yield inner_batch_cm


class StubCallbackContextManager:
    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0
        self.callback_spy = typed_async_mock(stub_callback)

    async def __aenter__(self):
        self.enter_count += 1
        return self.callback_spy

    async def __aexit__(self, *args):
        self.exit_count += 1


class StubBatchCallbackContextManager:
    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0
        self.callback_spy = typed_async_mock(stub_batch_callback)

    async def __aenter__(self):
        self.enter_count += 1
        return self.callback_spy

    async def __aexit__(self, *args):
        self.exit_count += 1


@pytest.fixture
def mock_send_batch_to[T]():
    with patch('mini.hither.send_batch_to', autospec=True) as mock:
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


async def test_run_hither_factory(mock_send_batch_to):
    _, _, mock_send_batch = mock_send_batch_to

    factory_mock = MagicMock(return_value=stub_callback)

    async with _run_hither_factory(factory_mock) as callback:
        callback(1, 'test')

    factory_mock.assert_called_once()
    mock_send_batch.assert_called_once()


async def test_run_hither_batch(mock_send_batch_to):
    _, _, mock_send_batch = mock_send_batch_to

    async with _run_hither_batch(stub_batch_callback) as callback:
        callback([1, 2, 3])

    mock_send_batch.assert_called_once_with([1, 2, 3])


async def test_run_hither_batch_factory(mock_send_batch_to):
    _, _, mock_send_batch = mock_send_batch_to

    factory_mock = MagicMock(return_value=stub_batch_callback)

    async with _run_hither_batch_factory(factory_mock) as callback:
        callback([1, 2, 3])

    factory_mock.assert_called_once()
    mock_send_batch.assert_called_once_with([1, 2, 3])


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
        mock_send_batch.assert_called_once()

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
        callback([1, 2, 3])

        # Verify the send_batch was called with correct parameters
        mock_send_batch.assert_called_once_with([1, 2, 3])

    # After context exits, __aexit__ should be called
    assert cm.exit_count == 1, 'Context manager __aexit__ should be called after exiting the context'


async def test_run_hither_cm_factory(mock_send_batch_to):
    _, _, mock_send_batch = mock_send_batch_to

    # The factory returns the context manager
    factory_mock = MagicMock(return_value=StubCallbackContextManager())

    async with _run_hither_cm_factory(factory_mock) as callback:
        callback(1, 'test')

    factory_mock.assert_called_once()
    mock_send_batch.assert_called_once()


async def test_run_hither_batch_cm_factory(mock_send_batch_to):
    _, _, mock_send_batch = mock_send_batch_to

    # The factory returns the batch context manager
    factory_mock = MagicMock(return_value=StubBatchCallbackContextManager())

    async with _run_hither_batch_cm_factory(factory_mock) as callback:
        callback([1, 2, 3])

    factory_mock.assert_called_once()
    mock_send_batch.assert_called_once_with([1, 2, 3])


# Test the main run_hither function's auto-detection
async def test_run_hither_autodetection(mock_send_batch_to):
    # Test auto-detection of regular callback
    with patch('mini.hither._run_hither') as mock_run:
        mock_run.return_value.__aenter__.return_value = 'test'

        async with run_hither(stub_callback) as _:
            pass

        mock_run.assert_called_once_with(stub_callback)

    # Test auto-detection of context manager
    with patch('mini.hither._run_hither_cm') as mock_run_cm:
        mock_run_cm.return_value.__aenter__.return_value = 'test'
        cm = StubCallbackContextManager()

        async with run_hither(cm) as _:
            pass

        mock_run_cm.assert_called_once_with(cm)

    # Test auto-detection of context manager factory
    with patch('mini.hither._run_hither_cm_factory') as mock_run_cm_factory:
        mock_run_cm_factory.return_value.__aenter__.return_value = 'test'

        async with run_hither(stub_cb_cm) as _:
            pass

        mock_run_cm_factory.assert_called_once_with(stub_cb_cm)


async def test_run_hither_mode_specification(mock_send_batch_to):
    # Test explicit 'callback' mode
    with patch('mini.hither._run_hither') as mock_run:
        mock_run.return_value.__aenter__.return_value = 'test'

        async with run_hither(stub_callback, mode='callback') as _:
            pass

        mock_run.assert_called_once_with(stub_callback)

    # Test 'factory' mode
    with patch('mini.hither._run_hither_factory') as mock_run_factory:
        mock_run_factory.return_value.__aenter__.return_value = 'test'

        async with run_hither(stub_cb_factory, mode='factory') as _:
            pass

        mock_run_factory.assert_called_once_with(stub_cb_factory)

    # Test 'cm' mode
    with patch('mini.hither._run_hither_cm') as mock_run_cm:
        mock_run_cm.return_value.__aenter__.return_value = 'test'
        cm = StubCallbackContextManager()

        async with run_hither(cm, mode='cm') as _:
            pass

        mock_run_cm.assert_called_once_with(cm)

    # Test 'cm_factory' mode
    with patch('mini.hither._run_hither_cm_factory') as mock_run_cm_factory:
        mock_run_cm_factory.return_value.__aenter__.return_value = 'test'

        async with run_hither(stub_cb_cm, mode='cm_factory') as _:
            pass

        mock_run_cm_factory.assert_called_once_with(stub_cb_cm)


async def test_run_hither_batch_mode(mock_send_batch_to):
    # Test batch callback
    with patch('mini.hither._run_hither_batch') as mock_run_batch:
        mock_run_batch.return_value.__aenter__.return_value = 'test'

        async with run_hither(stub_batch_callback, batch=True) as _:
            pass

        mock_run_batch.assert_called_once_with(stub_batch_callback)

    # Test batch factory
    with patch('mini.hither._run_hither_batch_factory') as mock_run_batch_factory:
        mock_run_batch_factory.return_value.__aenter__.return_value = 'test'

        async with run_hither(stub_batch_cb_factory, batch=True, mode='factory') as _:
            pass

        mock_run_batch_factory.assert_called_once_with(stub_batch_cb_factory)

    # Test batch context manager
    with patch('mini.hither._run_hither_batch_cm') as mock_run_batch_cm:
        mock_run_batch_cm.return_value.__aenter__.return_value = 'test'
        cm = StubBatchCallbackContextManager()

        async with run_hither(cm, batch=True) as _:
            pass

        mock_run_batch_cm.assert_called_once_with(cm)

    # Test batch context manager factory
    with patch('mini.hither._run_hither_batch_cm_factory') as mock_run_batch_cm_factory:
        mock_run_batch_cm_factory.return_value.__aenter__.return_value = 'test'

        async with run_hither(stub_batch_cb_cm, batch=True) as _:
            pass

        mock_run_batch_cm_factory.assert_called_once_with(stub_batch_cb_cm)


async def test_run_hither_error_handling():
    # Test invalid mode
    with pytest.raises(ValueError, match='Invalid mode: invalid_mode'):
        async with run_hither(stub_callback, mode='invalid_mode'):  # type: ignore
            pass

    with pytest.raises(ValueError, match='Invalid mode: invalid_mode'):
        async with run_hither(stub_batch_callback, batch=True, mode='invalid_mode'):  # type: ignore
            pass


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


async def test_decorated_callbacks_preserve_metadata(mock_send_batch_to):
    """Test that metadata like __name__, __doc__ etc. are preserved."""
    async with _run_hither(stub_callback) as send_single:
        # Check function metadata is preserved
        assert send_single.__name__ == stub_callback.__name__
        assert send_single.__doc__ == stub_callback.__doc__
        assert send_single.__module__ == stub_callback.__module__

    async def test_run_hither_cm_fixed(mock_send_batch_to):
        """Test that context managers are correctly handled with proper implementation."""
        mock_fn, _, mock_send_batch = mock_send_batch_to

        # Create a real context manager with proper instrumentation
        cm = StubCallbackContextManager()

        # Use a spy to track calls to the CM's methods
        enter_spy = AsyncMock()
        exit_spy = AsyncMock()

        # Save original methods before monkey patching
        original_enter = cm.__aenter__
        original_exit = cm.__aexit__

        # Create wrapped versions that call both original and spy
        async def wrapped_enter(*args, **kwargs):
            await enter_spy(*args, **kwargs)
            return await original_enter(*args, **kwargs)

        async def wrapped_exit(*args, **kwargs):
            await exit_spy(*args, **kwargs)
            return await original_exit(*args, **kwargs)

        # Apply the monkey patches
        cm.__aenter__ = wrapped_enter
        cm.__aexit__ = wrapped_exit

        # Run the test - use our implementation directly
        async with _run_hither_cm(cm) as callback:
            # Check that __aenter__ was called immediately when entering the context
            assert enter_spy.call_count == 1, "Context manager's __aenter__ should be called when creating the context"
            assert exit_spy.call_count == 0, "Context manager's __aexit__ should not be called until context exit"
            callback(1, 'test')

            # Verify send_batch was called with appropriate parameters
            mock_fn.assert_called_once()

        # After context is exited, __aexit__ should have been called
        assert exit_spy.call_count == 1, "Context manager's __aexit__ should be called after context exit"

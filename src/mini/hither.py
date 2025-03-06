import logging
from contextlib import AbstractAsyncContextManager, AsyncContextDecorator, asynccontextmanager
from functools import wraps
from typing import Any, AsyncGenerator, Callable, ParamSpec, TypeAlias, TypeVar, overload

from mini._mode_detect import detect_mode
from mini.local_dispatch import send_batch_to
from mini.types import AsyncCallable, Params

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')


log = logging.getLogger(__name__)


Factory: TypeAlias = Callable[[], R]
Callback: TypeAlias = Callable[P, Any | None]
AsyncCallback: TypeAlias = AsyncCallable[P, Any | None]
CallbackContextManager: TypeAlias = AbstractAsyncContextManager[Callback[P]]
AsyncCallbackContextManager: TypeAlias = AbstractAsyncContextManager[AsyncCallback[P]]

# @asynccontextmanager needs *both* of these types to match
AsyncCallbackContextDecorator: TypeAlias = Factory[AsyncContextDecorator | AsyncCallbackContextManager[P]]
"""Functions decorated with @asynccontextmanager (they're factories that return CMs)"""

AsyncBatchCallback: TypeAlias = AsyncCallback[[list[T]]]
AsyncBatchCallbackContextManager: TypeAlias = AbstractAsyncContextManager[AsyncBatchCallback[T]]

# @asynccontextmanager needs *both* of these types to match
AsyncBatchCallbackContextDecorator: TypeAlias = Factory[AsyncContextDecorator | AsyncBatchCallbackContextManager[T]]
"""Functions decorated with @asynccontextmanager (they're factories that return CMs)"""

# These function decorators cause a function to always run locally, even when called in a remote Modal worker.

# Batching types:
# - One-to-one:  one invocation of the local callback per remote call  (*args, **kwargs) -> None
# - Many-to-one: one invocation per batch of values received           (list[T]) -> None
#
# Constructor types:
# - Instance
# - Factory
#
# Lifecycle:
# - Unmanaged
# - Context manager


# Non-batch overloads
@overload
def run_hither(callback: AsyncCallback[P]) -> CallbackContextManager[P]: ...


@overload
def run_hither(callback: Factory[AsyncCallback[P]]) -> Factory[CallbackContextManager[P]]: ...


@overload
def run_hither(callback: AsyncCallbackContextManager[P]) -> CallbackContextManager[P]: ...


@overload
def run_hither(callback: AsyncCallbackContextDecorator[P]) -> Factory[CallbackContextManager[P]]: ...


def run_hither(callback):  # type: ignore
    """
    Run a callback locally, even when called in a remote Modal worker.

    Args:
        callback: The callback to run locally (see below)

    - An **async** function (bare callback)
    - A regular function that returns an **async** function.
    - An async context manager that yields an **async** function
    - An `@asynccontextmanager` that yields an **async** callback

    Returns:
        stub:
        A function that takes the same parameters as `callback`, but which just puts the request on a queue and returns immediately. It returns `None`, even if `callback` returns something else.
    """
    mode = detect_mode(callback)
    if mode == 'callback':
        return _run_hither(callback)
    elif mode == 'factory':
        # return a function that instantiates the callback and wraps it in our context manager
        return lambda *args, **kwargs: _run_hither(callback(*args, **kwargs))
    elif mode == 'cm':
        return _run_hither_cm(callback)
    elif mode == 'cm_factory':
        # return a function that instantiates the context manager and wraps it in *our* context manager
        return lambda *args, **kwargs: _run_hither_cm(callback(*args, **kwargs))
    else:
        raise ValueError(f'Invalid mode: {mode}')


# Batch overloads
@overload
def run_hither_batch(callback: AsyncBatchCallback[T]) -> CallbackContextManager[T]: ...


@overload
def run_hither_batch(callback: Factory[AsyncBatchCallback[T]]) -> Factory[CallbackContextManager[T]]: ...


@overload
def run_hither_batch(callback: AsyncBatchCallbackContextManager[T]) -> CallbackContextManager[T]: ...


@overload
def run_hither_batch(callback: AsyncBatchCallbackContextDecorator[T]) -> Factory[CallbackContextManager[T]]: ...


def run_hither_batch(callback):  # type: ignore
    """
    Run a batched callback locally, even when called in a remote Modal worker.

    The callback can be:

    - An **async** function (bare callback)
    - A regular function that returns an **async** function.
    - An async context manager that yields an **async** function
    - An `@asynccontextmanager` that yields an **async** callback

    Args:
        callback: The callback to run locally (see below). It must take only one parameter: a list of objects `T`.

    Returns:
        unbatched-stub:
        A function that takes the same parameter type `T` as `callback`, but which just puts the request on a queue and returns immediately. It returns `None`, even if `callback` returns something else. Even though this stub function takes values one at a time, the callback may be called with more items (depending on how quickly they're read from the queue).
    """
    mode = detect_mode(callback)
    if mode == 'callback':
        return _run_hither_batch(callback)
    elif mode == 'factory':
        # return a function that instantiates the callback and wraps it in our context manager
        return lambda *args, **kwargs: _run_hither_batch(callback(*args, **kwargs))
    elif mode == 'cm':
        return _run_hither_batch_cm(callback)
    elif mode == 'cm_factory':
        # return a function that instantiates the context manager and wraps it in *our* context manager
        return lambda *args, **kwargs: _run_hither_batch_cm(callback(*args, **kwargs))
    else:
        raise ValueError(f'Invalid mode: {mode}')


@asynccontextmanager
async def _run_hither(
    callback: AsyncCallback[P],
) -> AsyncGenerator[Callback[P]]:
    @wraps(callback, assigned=('__module__', '__name__', '__qualname__', '__doc__'))
    async def batched_callback(calls: list[Params[P]]) -> None:
        for call in calls:
            await callback(*call.args, **call.kwargs)

    log.debug('Starting producer and consumer for %s', callback)
    async with send_batch_to(batched_callback) as send_batch:

        @wraps(callback)
        def send_single(*args, **kwargs):
            send_batch([Params[P](args, kwargs)])

        # Remove the reference to the wrapped function so it doesn't get serialized.
        del send_single.__wrapped__

        yield send_single


@asynccontextmanager
async def _run_hither_batch(
    callback: AsyncCallback[list[T]],
) -> AsyncGenerator[Callback[T]]:
    log.debug('Starting batched producer and consumer for %s', callback)
    async with send_batch_to(callback) as send_batch:

        @wraps(callback)
        def send_single(value: T):
            send_batch([value])

        # Remove the reference to the wrapped function so it doesn't get serialized.
        del send_single.__wrapped__

        yield send_single


@asynccontextmanager
async def _run_hither_cm(
    cb_context: AsyncCallbackContextManager[P],
) -> AsyncGenerator[Callback[P]]:
    log.debug('Entering callback context %s', cb_context)
    async with cb_context as callback:
        async with _run_hither(callback) as send:
            yield send


@asynccontextmanager
async def _run_hither_batch_cm(
    cb_context: AsyncCallbackContextManager[list[T]],
) -> AsyncGenerator[Callback[T]]:
    log.debug('Entering batched callback context %s', cb_context)
    async with cb_context as callback:
        async with _run_hither_batch(callback) as send:
            yield send


# if __name__ == '__main__':
#     # Example usage

#     # 1. A plain function

#     async def plain_cb(x: int, y: float):
#         print(f'Callback called with {x},{y}')

#     # 2. A factory function

#     def factory_callback():
#         print('Before')

#         async def inner(x: int, y: float):
#             print(f'Inner callback called with {x},{y}')

#         return inner

#     # 3. A batch function

#     async def batch_callback(xs: list[int]):
#         print(f'Batch callback called with {xs}')

#     # 4. A factory function that returns a batch function

#     def batch_factory_callback():
#         print('Before')

#         async def inner(xs: list[int]):
#             print(f'Inner batch callback called with {xs}')

#         return inner

#     # 5. A context manager

#     @asynccontextmanager
#     async def context_callback():
#         print('Before')

#         async def inner(x: int, y: float):
#             print(f'Inner batch callback called with {x},{y}')

#         yield inner
#         print('After')

#     # 6. A context manager that returns a batch function

#     @asynccontextmanager
#     async def context_batch_callback():
#         print('Before')

#         async def inner(xs: list[int]):
#             print(f'Inner batch callback called with {xs}')

#         yield inner
#         print('After')

#     async def main():
#         async with (
#             _run_hither(plain_cb) as _plain_cb,
#             _run_hither_factory(factory_callback) as _factory_cb,
#             _run_hither_batch(batch_callback) as _batch_cb,
#             _run_hither_batch_factory(batch_factory_callback) as _batch_factory_cb,
#             _run_hither_cm_factory(context_callback) as _context_cb,
#             _run_hither_batch_cm_factory(context_batch_callback) as _context_batch_cb,
#         ):
#             # These are processed locally one at a time (but in parallel)
#             callbacks: Sequence[Callback[[int, float]]] = [
#                 _plain_cb,
#                 _factory_cb,
#                 _context_cb,
#             ]
#             # These are processed in batches
#             batch_callbacks: Sequence[Callback[[list[int]]]] = [
#                 _batch_cb,
#                 _batch_factory_cb,
#                 _context_batch_cb,
#             ]
#             await train.remote.aio(callbacks, batch_callbacks)

#         async with (
#             run_hither(plain_cb) as _plain_cb,
#             run_hither(factory_callback, mode='factory') as _factory_cb,
#             run_hither(batch_callback, batch=True) as _batch_cb,
#             run_hither(batch_factory_callback, batch=True, mode='factory') as _batch_factory_cb,
#             run_hither(context_callback) as _context_cb,
#             run_hither(context_batch_callback, batch=True) as _context_batch_cb,
#         ):
#             # These are processed locally one at a time (but in parallel)
#             callbacks: Sequence[Callback[[int, float]]] = [
#                 _plain_cb,
#                 _factory_cb,
#                 _context_cb,
#             ]
#             # These are processed in batches
#             batch_callbacks: Sequence[Callback[[list[int]]]] = [
#                 _batch_cb,
#                 _batch_factory_cb,
#                 _context_batch_cb,
#             ]
#             await train.remote.aio(callbacks, batch_callbacks)

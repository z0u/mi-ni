import inspect
import logging
from contextlib import AbstractAsyncContextManager, AsyncContextDecorator, asynccontextmanager
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Literal, ParamSpec, TypeAlias, TypeVar, overload

from mini.local_dispatch import send_batch_to
from mini.types import AsyncCallable, Params

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')


log = logging.getLogger(__name__)


Factory: TypeAlias = Callable[[], R]
Callback: TypeAlias = Callable[P, Any | None]
AsyncCallback: TypeAlias = AsyncCallable[
    P, Any | None
]  # Callable[P, Awaitable[None]] | Callable[P, CoroutineType[Any, Any, None]]
CallbackContextManager: TypeAlias = AbstractAsyncContextManager[Callback[P]]
AsyncCallbackContextManager: TypeAlias = AbstractAsyncContextManager[AsyncCallback[P]]

BatchCallback: TypeAlias = Callback[[list[T]]]
AsyncBatchCallback: TypeAlias = AsyncCallback[[list[T]]]
BatchCallbackContextManager: TypeAlias = AbstractAsyncContextManager[BatchCallback[T]]
AsyncBatchCallbackContextManager: TypeAlias = AbstractAsyncContextManager[AsyncBatchCallback[T]]

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
def run_hither(
    callback: AsyncCallback[P],
    *,
    batch: Literal[False] = False,
    mode: Literal[None, 'callback'] = None,
) -> CallbackContextManager[P]: ...


@overload
def run_hither(
    callback: Factory[AsyncCallback[P]],
    *,
    batch: Literal[False] = False,
    mode: Literal['factory'],
) -> CallbackContextManager[P]: ...


@overload
def run_hither(
    callback: AsyncCallbackContextManager[P],
    *,
    batch: Literal[False] = False,
    mode: Literal[None, 'cm'] = None,
) -> CallbackContextManager[P]: ...


# For functions decorated with @asynccontextmanager (they're factories that return CMs)
@overload
def run_hither(
    # @asynccontextmanager needs *both* of these types to match
    callback: Factory[AsyncContextDecorator | AsyncCallbackContextManager[P]],
    *,
    batch: Literal[False] = False,
    mode: Literal[None, 'cm_factory'] = None,
) -> CallbackContextManager[P]: ...


# Batch overloads
@overload
def run_hither(
    callback: AsyncBatchCallback[T],
    *,
    batch: Literal[True],
    mode: Literal[None, 'callback'] = None,
) -> BatchCallbackContextManager[T]: ...


@overload
def run_hither(
    callback: Factory[AsyncBatchCallback[T]],
    *,
    batch: Literal[True],
    mode: Literal['factory'],
) -> BatchCallbackContextManager[T]: ...


@overload
def run_hither(
    callback: AsyncBatchCallbackContextManager[T],
    *,
    batch: Literal[True],
    mode: Literal[None, 'cm'] = None,
) -> BatchCallbackContextManager[T]: ...


# And for the batch version:
@overload
def run_hither(
    # @asynccontextmanager needs *both* of these types to match
    callback: Factory[AsyncContextDecorator | AsyncBatchCallbackContextManager[T]],
    *,
    batch: Literal[True],
    mode: Literal[None, 'cm_factory'] = None,
) -> BatchCallbackContextManager[T]: ...


def run_hither(callback, *, batch=False, mode=None):  # type: ignore[no-untyped-def]  # noqa: C901
    """
    Run a callback locally, even when called in a remote Modal worker.

    Args:
        callback: The callback to run locally.
        batch: Whether the callback takes a batch of inputs.
        mode: The kind of callback to run. This can be one of the following:
            - None: Auto-detect based on the callback type (default)
            - 'callback': Treat as a regular callback
            - 'factory': Treat as a factory function that returns a callback (can't be auto-detected)
            - 'cm': Treat as a context manager
            - 'cm_factory': Treat as a factory that returns a context manager
    """
    # Auto-detect the callback type
    if mode is None:
        if hasattr(callback, '__aenter__') and hasattr(callback, '__aexit__'):
            # It has context manager methods directly - it's a CM instance
            mode = 'cm'
        elif inspect.isasyncgenfunction(callback):
            # It's a function decorated with @asynccontextmanager
            # This is actually a factory that returns a context manager when called
            mode = 'cm_factory'
        else:
            # Default assumption: it's a regular callback
            mode = 'callback'

    # Dispatch based on mode and batch flag
    if batch:
        if mode == 'callback':
            return _run_hither_batch(callback)
        elif mode == 'factory':
            return _run_hither_batch_factory(callback)
        elif mode == 'cm':
            return _run_hither_batch_cm(callback)
        elif mode == 'cm_factory':
            return _run_hither_batch_cm_factory(callback)
        else:
            raise ValueError(f'Invalid mode: {mode}')
    else:
        if mode == 'callback':
            return _run_hither(callback)
        elif mode == 'factory':
            return _run_hither_factory(callback)
        elif mode == 'cm':
            return _run_hither_cm(callback)
        elif mode == 'cm_factory':
            return _run_hither_cm_factory(callback)
        else:
            raise ValueError(f'Invalid mode: {mode}')


@asynccontextmanager
async def _run_hither(
    callback: AsyncCallback[P],
) -> AsyncGenerator[Callback[P]]:
    # Batch and unbatch because `send_batch_to` expects a list of items
    cb = _unbatch(callback)
    async with send_batch_to(cb) as send_batch:
        send_batches_of_one = _batches_of_one(send_batch)
        yield send_batches_of_one


@asynccontextmanager
async def _run_hither_factory(
    cb_factory: Factory[AsyncCallback[P]],
) -> AsyncGenerator[Callback[P]]:
    async with _run_hither(cb_factory()) as send:
        yield send


@asynccontextmanager
async def _run_hither_batch(
    callback: AsyncCallback[list[T]],
) -> AsyncGenerator[Callback[list[T]]]:
    async with send_batch_to(callback) as send_batch:
        yield send_batch


@asynccontextmanager
async def _run_hither_batch_factory(
    callback: Factory[AsyncCallback[list[T]]],
) -> AsyncGenerator[Callback[list[T]]]:
    async with _run_hither_batch(callback()) as send_batch:
        yield send_batch


@asynccontextmanager
async def _run_hither_cm(
    cb_context: AsyncCallbackContextManager[P],
) -> AsyncGenerator[Callback[P]]:
    async with cb_context as callback, _run_hither(callback) as send:
        yield send


@asynccontextmanager
async def _run_hither_batch_cm(
    cb_context: AsyncCallbackContextManager[list[T]],
) -> AsyncGenerator[Callback[list[T]]]:
    async with cb_context as callback, _run_hither_batch(callback) as send:
        yield send


@asynccontextmanager
async def _run_hither_cm_factory(
    cb_context_factory: Factory[AsyncCallbackContextManager[P]],
) -> AsyncGenerator[Callback[P]]:
    async with _run_hither_cm(cb_context_factory()) as send:
        yield send


@asynccontextmanager
async def _run_hither_batch_cm_factory(
    cb_context_factory: Factory[AsyncCallbackContextManager[list[T]]],
) -> AsyncGenerator[Callback[list[T]]]:
    async with cb_context_factory() as callback:
        async with _run_hither_batch(callback) as send:
            yield send


def _batches_of_one(fn: Callable[[list[Params[P]]], Any]) -> Callable[P, Any]:
    @wraps(fn, assigned=('__module__', '__name__', '__qualname__', '__doc__'))
    def wrapper(*args, **kwargs):
        params = Params[P](*args, **kwargs)
        return fn([params])

    return wrapper  # type: ignore[no-redef]


def _unbatch(fn: AsyncCallable[P, Any]) -> AsyncCallable[[list[Params[P]]], None]:
    @wraps(fn, assigned=('__module__', '__name__', '__qualname__', '__doc__'))
    async def wrapper(calls: list[Params[P]]) -> None:
        for call in calls:
            await fn(*call.args, **call.kwargs)

    return wrapper  # type: ignore[no-redef]


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

import inspect
from functools import wraps
from typing import Awaitable, Callable, ParamSpec, TypeVar, cast

P = ParamSpec('P')
R = TypeVar('R')


def coerce_to_async(fn: Callable[P, R | Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    if inspect.iscoroutinefunction(fn):
        return fn

    fn = cast(Callable[P, R], fn)

    @wraps(fn)
    async def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper

import typing
from dataclasses import dataclass
from typing import Awaitable, Callable, ParamSpec, Protocol, TypeAlias, TypeVar, runtime_checkable

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')


SyncHandler: TypeAlias = Callable[[T], None]
AsyncHandler: TypeAlias = Callable[[T], Awaitable[None]]
Handler: TypeAlias = SyncHandler[T] | AsyncHandler[T]


@runtime_checkable
class AsyncCallable(Protocol[P, R]):
    """Represents an async callable specifically."""

    __call__: Callable[P, Awaitable[R]]
    __name__: str
    __module__: str
    __qualname__: str
    __annotations__: dict
    __doc__: str | None


# A single Queue can contain [...] up to 5,000 items.
# https://modal.com/docs/reference/modal.Queue
Q_MAX_LEN = 5_000


@dataclass
class Params(typing.Generic[P]):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

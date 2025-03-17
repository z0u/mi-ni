from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, ParamSpec, Protocol, TypeAlias, TypeVar, runtime_checkable

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


FnId: TypeAlias = tuple[str, str]
Partition: TypeAlias = str | None


@dataclass
class Call:
    """cloudpickle-friendly representation of a function call."""

    fn_id: FnId
    args: tuple
    kwargs: Mapping

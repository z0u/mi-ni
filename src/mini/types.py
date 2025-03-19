import typing
from dataclasses import dataclass
from typing import Awaitable, Callable, ParamSpec, TypeAlias, TypeVar

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')

AsyncCallable: TypeAlias = Callable[P, Awaitable[R]]
SyncHandler: TypeAlias = Callable[[T], None]
AsyncHandler: TypeAlias = AsyncCallable[[T], None]
Handler: TypeAlias = SyncHandler[T] | AsyncHandler[T]


# A single Queue can contain [...] up to 5,000 items.
# https://modal.com/docs/reference/modal.Queue
Q_MAX_LEN = 5_000


@dataclass
class Params(typing.Generic[P]):
    args: tuple
    kwargs: dict

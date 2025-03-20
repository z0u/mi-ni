from contextlib import AbstractContextManager
from types import TracebackType
import typing
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, ParamSpec, Type, TypeAlias, TypeVar, Union

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')

AsyncCallable: TypeAlias = Callable[P, Awaitable[R]]
SyncHandler: TypeAlias = Callable[[T], None]
AsyncHandler: TypeAlias = AsyncCallable[[T], None]
Handler: TypeAlias = SyncHandler[T] | AsyncHandler[T]
Ignored: TypeAlias = object | None

ExceptionInfo: TypeAlias = tuple[
    Optional[Type[BaseException]],
    Optional[BaseException],
    Optional[TracebackType],
]
RemoteFunction: TypeAlias = AsyncCallable[P, R]

Guard: TypeAlias = Callable[[], None]
GuardFn: TypeAlias = Callable[[RemoteFunction[P, Ignored]], None]
GuardExc: TypeAlias = Callable[[*ExceptionInfo], None]
GuardFnExc: TypeAlias = Callable[[RemoteFunction[P, Ignored], *ExceptionInfo], None]

GuardContext: TypeAlias = Callable[[], AbstractContextManager]
GuardContextFn: TypeAlias = Callable[[RemoteFunction[P, Ignored]], AbstractContextManager]

GuardDecorator: TypeAlias = Union[
    Callable[[GuardContext], GuardContext],
    Callable[[GuardContextFn[P]], GuardContextFn[P]],
]
"""Function decorator for guards (before & after). These must already be context manager functions, e.g. decorated with `@contextmanager`."""

BeforeGuardDecorator: TypeAlias = Union[
    Callable[[Guard], GuardContext],
    Callable[[GuardFn[P]], GuardContextFn[P]],
]
"""Function decorator for guards that run before function execution."""

AfterGuardDecorator: TypeAlias = Union[
    Callable[[Guard | GuardExc], GuardContext],
    Callable[[GuardFn[P] | GuardFnExc[P]], GuardContextFn[P]],
]
"""Function decorator for guards that run after function execution. These can also be used to process exceptions."""


# A single Queue can contain [...] up to 5,000 items.
# https://modal.com/docs/reference/modal.Queue
Q_MAX_LEN = 5_000


@dataclass
class Params(typing.Generic[P]):
    args: tuple
    kwargs: dict

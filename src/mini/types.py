import typing
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Awaitable, Callable, Optional, ParamSpec, Type, TypeAlias, TypeVar, Union

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')

AsyncCallable: TypeAlias = Callable[P, Awaitable[R]]
SyncHandler: TypeAlias = Callable[[T], None]
AsyncHandler: TypeAlias = AsyncCallable[[T], None]
Handler: TypeAlias = SyncHandler[T] | AsyncHandler[T]

ExceptionInfo: TypeAlias = tuple[
    Optional[Type[BaseException]],
    Optional[BaseException],
    Optional[TracebackType],
]
RemoteFunction: TypeAlias = AsyncCallable[P, R]

Guard: TypeAlias = Callable[[], R]
GuardFn: TypeAlias = Callable[[RemoteFunction[P, Any]], R]


GuardExc: TypeAlias = Callable[[*ExceptionInfo], R]
GuardFnExc: TypeAlias = Callable[[RemoteFunction[P, Any], *ExceptionInfo], R]


GuardContext: TypeAlias = Callable[[], AbstractContextManager]
GuardContextFn: TypeAlias = Callable[[RemoteFunction[P, Any]], AbstractContextManager]


GuardDecorator: TypeAlias = Union[
    Callable[[GuardContext], GuardContext],
    Callable[[GuardContextFn[P]], GuardContextFn[P]],
]
"""Function decorator for guards (before & after). These must already be context manager functions, e.g. decorated with `@contextmanager`."""

BeforeGuardDecorator: TypeAlias = Callable[[Guard[R]], GuardContext]
"""Function decorator for guards that run before function execution."""
BeforeGuardDecoratorFn: TypeAlias = Callable[[GuardFn[P, R]], GuardContextFn[P]]
"""Function decorator for guards that run before function execution."""


AfterGuardDecorator: TypeAlias = Callable[[Guard[R] | GuardExc[R]], GuardContext]
"""Function decorator for guards that run after function execution. These can also be used to process exceptions."""
AfterGuardDecoratorFn: TypeAlias = Callable[[GuardFn[P, R] | GuardFnExc[P, R]], GuardContextFn[P]]
"""Function decorator for guards that run after function execution. These can also be used to process exceptions."""


# A single Queue can contain [...] up to 5,000 items.
# https://modal.com/docs/reference/modal.Queue
Q_MAX_LEN = 5_000


@dataclass
class Params(typing.Generic[P]):
    args: tuple
    kwargs: dict

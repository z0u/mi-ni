import inspect
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar, cast
from typing import overload

from mini.types import ExceptionInfo, Guard, GuardContext, GuardContextFn, GuardExc, GuardFn, GuardFnExc, RemoteFunction

P = ParamSpec('P')
R = TypeVar('R')


@overload
def before(callback: Guard[R]) -> GuardContext: ...
@overload
def before(callback: GuardFn[P, R]) -> GuardContextFn[P]: ...


def before(callback: Guard[R] | GuardFn[P, R]) -> GuardContext | GuardContextFn[P]:
    """Create a guard that runs before the function is executed."""
    if len(inspect.signature(callback).parameters) == 0:
        callback = cast(Guard[R], callback)

        @contextmanager
        @wraps(callback)
        def guard():
            callback()
            yield

        return guard
    else:
        callback = cast(GuardFn[P, R], callback)

        @contextmanager
        @wraps(callback)
        def guard_fn(fn: RemoteFunction[P, R]):
            callback(fn)
            yield

        return guard_fn


@overload
def after(callback: Guard[R]) -> GuardContext: ...
@overload
def after(callback: GuardFn[P, R]) -> GuardContextFn[P]: ...
@overload
def after(callback: GuardExc[R]) -> GuardContext: ...
@overload
def after(callback: GuardFnExc[P, R]) -> GuardContextFn[P]: ...


def after(callback: Guard[R] | GuardFn[P, R] | GuardExc[R] | GuardFnExc[P, R]) -> GuardContext | GuardContextFn[P]:
    """Create a guard that runs after the function is executed."""
    if len(inspect.signature(callback).parameters) == 0:
        callback = cast(Guard[R], callback)

        @contextmanager
        @wraps(callback)
        def guard():
            try:
                yield
            finally:
                callback()

        return guard

    elif len(inspect.signature(callback).parameters) == 1:
        callback = cast(GuardFn[P, R], callback)

        @contextmanager
        @wraps(callback)
        def guard_fn(fn: RemoteFunction[P, R]):
            try:
                yield
            finally:
                callback(fn)

        return guard_fn

    elif len(inspect.signature(callback).parameters) == 3:
        callback = cast(GuardExc, callback)

        @contextmanager
        @wraps(callback)
        def guard_exc():
            ex_info: ExceptionInfo = (None, None, None)
            try:
                yield
            except BaseException as e:
                ex_info = (type(e), e, e.__traceback__)
            finally:
                callback(*ex_info)

        return guard_exc

    else:
        callback = cast(GuardFnExc[P, R], callback)

        @contextmanager
        @wraps(callback)
        def guard_fn_exc(fn: RemoteFunction[P, R]):
            ex_info: ExceptionInfo = (None, None, None)
            try:
                yield
            except BaseException as e:
                ex_info = (type(e), e, e.__traceback__)
            finally:
                callback(fn, *ex_info)

        return guard_fn_exc

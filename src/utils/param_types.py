import inspect
from typing import Annotated, Any, Callable, TypeVar, overload

from annotated_types import Gt, Interval, MultipleOf, Predicate
from pydantic import ConfigDict
from pydantic import validate_call as _validate_call

__all__ = ['ZeroToOne', 'IntX8', 'IntX32', 'IntX64', 'PowerOf2', 'validate_call']

T = TypeVar('T')


ZeroToOne = Annotated[float, Interval(ge=0, le=1)]
"""Between 0 and 1 (inclusive)"""

IntX8 = Annotated[int, Gt(0), MultipleOf(8)]
"""Multiple of 8"""

IntX32 = Annotated[int, Gt(0), MultipleOf(32)]
"""Multiple of 32"""

IntX64 = Annotated[int, Gt(0), MultipleOf(64)]
"""Multiple of 64"""

PowerOf2 = Annotated[int, Gt(0), Predicate(lambda v: (v & (v - 1)) == 0)]
"""Power of 2"""

AnyCallableT = TypeVar('AnyCallableT', bound=Callable[..., Any])


@overload
def validate_call(
    *,
    config: ConfigDict | None = None,
    validate_return: bool | None = None,
) -> Callable[[AnyCallableT], AnyCallableT]: ...


@overload
def validate_call(func: AnyCallableT, /) -> AnyCallableT: ...


def validate_call(
    func: AnyCallableT | None = None,
    /,
    *,
    config: ConfigDict | None = None,
    validate_return: bool | None = None,
) -> AnyCallableT | Callable[[AnyCallableT], AnyCallableT]:
    """
    Validate the call signature of a function.

    Like `pydantic.validate_call`, but allows arbitrary parameter types by default.

    If `validate_return` is None (default), return value validation is automatically
    enabled if the function has a return type annotation, and disabled otherwise.

    Args:
        func: The function to validate
        config: Additional Pydantic configuration
        validate_return: Whether to validate return values (auto-detected if None)
    """
    _config = ConfigDict(arbitrary_types_allowed=True)
    if config is not None:
        _config.update(config)

    def wrapper(fn: AnyCallableT) -> AnyCallableT:
        nonlocal validate_return
        if validate_return is None:
            signature = inspect.signature(fn)
            validate_return = signature.return_annotation is not inspect.Signature.empty

        return _validate_call(config=_config, validate_return=validate_return)(fn)

    if func is not None:
        return wrapper(func)
    else:
        return wrapper

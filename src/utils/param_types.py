from typing import Annotated, Any, Callable, TypeVar, overload

from pydantic import AfterValidator, ConfigDict, PositiveInt
from pydantic import validate_call as _validate_call

__all__ = ["ZeroToOne", "IntX8", "IntX32", "IntX64", "PowerOf2", "validate_call"]

T = TypeVar("T")


def check(check_fn: Callable[[T], bool], error_msg: str) -> Callable[[T], T]:
    """Create a validator that returns the value if valid."""

    def validator(v: T) -> T:
        if not check_fn(v):
            raise ValueError(f"Value {v} {error_msg}")
        return v

    return validator


ZeroToOne = Annotated[
    float,
    AfterValidator(check(lambda v: 0 <= v <= 1, "must be between 0 and 1")),
]
"""Between 0 and 1 (inclusive)"""

IntX8 = Annotated[
    PositiveInt,
    AfterValidator(check(lambda v: v % 8 == 0, "must be a multiple of 8")),
]
"""Multiple of 8"""

IntX32 = Annotated[
    PositiveInt,
    AfterValidator(check(lambda v: v % 32 == 0, "must be a multiple of 32")),
]
"""Multiple of 32"""

IntX64 = Annotated[
    PositiveInt,
    AfterValidator(check(lambda v: v % 64 == 0, "must be a multiple of 64")),
]
"""Multiple of 64"""

PowerOf2 = Annotated[
    PositiveInt,
    AfterValidator(check(lambda v: (v & (v - 1)) == 0, "must be a power of 2")),
]
"""Power of 2"""


AnyCallableT = TypeVar("AnyCallableT", bound=Callable[..., Any])


@overload
def validate_call(
    *,
    config: ConfigDict | None = None,
    validate_return: bool = False,
) -> Callable[[AnyCallableT], AnyCallableT]: ...


@overload
def validate_call(func: AnyCallableT, /) -> AnyCallableT: ...


def validate_call(
    func: AnyCallableT | None = None,
    /,
    *,
    config: ConfigDict | None = None,
    validate_return: bool = False,
) -> AnyCallableT | Callable[[AnyCallableT], AnyCallableT]:
    """
    Validate the call signature of a function.

    Like `pydantic.validate_call`, but allows arbitrary parameter types by default.
    """
    _config = ConfigDict(arbitrary_types_allowed=True)
    if config is not None:
        _config.update(config)

    validate = _validate_call(config=_config, validate_return=validate_return)
    if func is not None:
        return validate(func)
    else:
        return validate

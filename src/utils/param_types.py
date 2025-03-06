from typing import Annotated
from pydantic import AfterValidator, PositiveInt

__all__ = ['ZeroToOne', 'IntX8', 'IntX32', 'IntX64', 'PowerOf2']

ZeroToOne = Annotated[
    float,
    AfterValidator(lambda v: 0 <= v <= 1),
]
"""Float between 0 and 1"""

IntX8 = Annotated[
    PositiveInt,
    AfterValidator(lambda v: v % 8 == 0),
]
"""Multiple of 8"""

IntX32 = Annotated[
    PositiveInt,
    AfterValidator(lambda v: v % 32 == 0),
]
"""Multiple of 32"""

IntX64 = Annotated[
    PositiveInt,
    AfterValidator(lambda v: v % 64 == 0),
]
"""Multiple of 64"""

PowerOf2 = Annotated[
    PositiveInt,
    AfterValidator(lambda v: (v & (v - 1)) == 0),
]
"""Power of 2"""

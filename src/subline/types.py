from dataclasses import dataclass
from math import isclose


@dataclass
class TokenBB:
    """
    1D bounding box for a token.

    All positions are **relative** to the token's starting position.
    """

    width: float
    """Total width of token"""

    first_char: float
    """Position of first char midpoint"""

    mid: float
    """Midpoint of token"""

    last_char: float
    """Position of last char midpoint"""

    @property
    def is_wide(self):
        return not isclose(self.first_char, self.last_char, rel_tol=0.05)

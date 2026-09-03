import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Series:
    """One line under the text: a value per token, plus how to draw it.

    *raw* is one value per token, in whatever units the caller holds them; :meth:`normalize` maps them to the band's 0–1 fractions, and the base class passes them through unchanged. So either scale before constructing, or subclass to carry the units with the series (:class:`EntropySeries`). ``NaN`` marks a position with no value and breaks the line there. *label* names the series in the legend, and is the only place it appears.
    """

    raw: np.ndarray
    color: str = ""
    dasharray: str = ""
    label: str = ""

    @property
    def values(self) -> np.ndarray:
        """Normalized values ready for visualization."""
        return self.normalize(self.raw)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Pass through raw values by default."""
        return x


@dataclass
class EntropySeries(Series):
    """A series that normalizes values relative to maximum possible entropy."""

    vocab_size: int = 256

    def normalize(self, x: np.ndarray) -> np.ndarray:
        max_entropy = math.log(self.vocab_size)
        return x / max_entropy

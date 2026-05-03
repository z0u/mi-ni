import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

DEFAULT_SPACE = 'linear'
DEFAULT_TF = 'minjerk'


@dataclass(slots=True)
class PropConfig:
    """Configuration for a single property column."""

    prop: str
    """Name of the property, e.g., 'x'"""
    space: str = DEFAULT_SPACE
    """Space type, e.g., 'log' or 'linear'."""
    timing_fn: str = DEFAULT_TF
    """Timing function to use for this property, e.g., 'linear', 'minjerk', etc."""

    @staticmethod
    def from_col_name(col: str):
        match = PROP_PATTERN.match(col)
        if match:
            parts = match.groupdict()
            prop_name = parts['prop']
            space = parts.get('space') or DEFAULT_SPACE
            interpolator_name = parts.get('interpolator') or DEFAULT_TF
            return PropConfig(prop=prop_name, space=space, timing_fn=interpolator_name)
        else:
            log.warning(f"Column '{col}' doesn't match 'prop:space:interpolator' format. Assuming defaults.")
            return PropConfig(prop=col)

    def __str__(self):
        timing_fn = self.timing_fn if self.timing_fn != DEFAULT_TF else ''
        space = self.space if self.space != DEFAULT_SPACE else ''
        if timing_fn:
            return f'{self.prop}:{space}:{timing_fn}'
        elif space:
            return f'{self.prop}:{space}'
        return f'{self.prop}'


PROP_PATTERN = re.compile(r'^(?P<prop>[^:]+)(?::(?P<space>[^:]*))?(?::(?P<interpolator>[^:]*))?$')


@dataclass(slots=True)
class Keyframe:
    """Metadata about a keyframe of a single property."""

    prop: str
    """Name of the property this keyframe is for."""
    t: int
    """The frame number."""
    value: float
    """The value at this step."""
    next_t: int | None
    """Frame number of the next keyframe for this property."""
    next_value: float | None
    """Value at the next keyframe for this property."""

    @property
    def duration(self) -> int | None:
        """Duration of the transition starting at this key."""
        if self.next_t is None:
            return None
        return self.next_t - self.t


@dataclass(slots=True)
class Frame:
    """Metadata about a single dopesheet frame."""

    t: int
    phase: str
    """The current phase active *at* this step."""
    is_phase_start: bool
    """Whether this is the first step of the phase."""
    is_phase_end: bool
    """Whether this is the last step of the phase."""
    actions: list[str]
    """Actions listed at this step."""
    keyed_props: list[Keyframe]
    """All properties that are keyed (have a non-NaN value) on this step."""


@dataclass(slots=True)
class DynamicPropState:
    value: float
    velocity: float
    acceleration: float


@dataclass(slots=True)
class TStep:
    """A timestep, with interpolated properties."""

    step: int
    phase: str
    actions: list[str]
    props: dict[str, float]
    is_phase_start: bool
    is_phase_end: bool

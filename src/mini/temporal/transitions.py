import math
from typing import Type

import numpy as np

from .model import DynamicPropState
from .timing_fn import MinimumJerkTimingFunction, TimingFunction


class DynamicProp:
    """Stores a value, and transitions new values based on a timing function."""

    _interpolator: TimingFunction | None
    _timing_function_cls: Type[TimingFunction]
    _ctime: float
    """The number of steps that have been taken since the last set() call."""
    _duration: float
    """The number of steps to take to reach the target value."""
    _value: float
    """The target value"""

    def __init__(
        self,
        value: float,
        duration: float = 0.0,
        timing_function_cls: Type[TimingFunction] = MinimumJerkTimingFunction,
    ):
        """
        Initialize the SmoothProp with a starting value and optional duration.

        Args:
            value: Initial value
            duration: Duration of the transition (unitless; see `step()`)
            timing_function_cls: The class to use for interpolation.
        """
        self._timing_function_cls = timing_function_cls
        self._duration = duration
        self._value = float(value)
        self._interpolator = None
        self._ctime = 0.0
        if not np.isclose(duration, 0.0):
            self._interpolator = self._timing_function_cls(
                initial_value=value,
                initial_velocity=0.0,
                initial_acceleration=0.0,
                final_value=value,
                duration=duration,
            )

    def step(self, n=1.0):
        """
        Progress the internal clock forward.

        Args:
            n: Amount to step the clock forward. This is a unitless value, and could
            mean "frames" or some real amount of time. For example, if called with the
            elapsed real time in seconds, the duration would be in seconds.
        """
        if self._interpolator is None:
            return

        self._ctime += float(n)
        if abs(self._ctime - self.duration) < 1e-10 or self._ctime > self.duration:
            final_val = self._interpolator.final_value
            self._interpolator = None
            self._value = final_val
            self._ctime = self.duration

    def set(self, value: float | None = None, duration: float | None = None):
        """
        Set a new target value and/or duration for the transition.

        Args:
            value: Target value to transition to. If `None`, keeps the current target value.
            duration: Duration of the transition. If `None`, keeps the current duration.

        Starts a new transition from the current state (value, velocity, acceleration)
        towards the new target value over the specified duration, using the configured
        timing function. Resets the internal clock `_ctime`.
        """
        new_target_value = (
            value if value is not None else (self._interpolator.final_value if self._interpolator else self._value)
        )
        new_duration = duration if duration is not None else self._duration

        # Get current state before potentially overwriting interpolator
        if self._interpolator is not None:
            # Ensure ctime is clamped before getting state if it overshot
            clamped_ctime = max(0.0, min(self._interpolator.duration, self._ctime))
            current_state = self._interpolator.get_state(clamped_ctime)  # Get TimingState object
        else:
            # If no interpolator, assume we are at rest at the last set value
            current_state = DynamicPropState(value=self._value, velocity=0.0, acceleration=0.0)  # Create TimingState

        # Handle immediate jump (zero duration)
        if np.isclose(new_duration, 0.0):
            self._value = float(new_target_value)
            self._interpolator = None
            self._ctime = 0.0
            self._duration = 0.0
            return

        # Avoid creating a new interpolator if the target hasn't changed
        # and we are already at rest at the target.
        if (
            self._interpolator is None  # Already at rest
            and np.isclose(current_state.value, new_target_value)  # Check value from state
        ):
            if not np.isclose(self._duration, new_duration):
                self._duration = float(new_duration)
            return  # No change needed

        # Avoid creating a new interpolator if target value and duration are identical to current transition
        # This prevents redundant object creation if set() is called repeatedly with the same target.
        if (
            self._interpolator is not None
            and np.isclose(self._interpolator.final_value, new_target_value)
            and np.isclose(self._interpolator.duration, new_duration)
        ):
            # Update the internal target value and duration in case they were None
            self._value = float(new_target_value)
            self._duration = float(new_duration)
            # Don't reset _ctime or create a new interpolator
            return

        # Create the new interpolator using the stored class
        self._interpolator = self._timing_function_cls(
            initial_value=current_state.value,  # Unpack from state
            initial_velocity=current_state.velocity,  # Unpack from state
            initial_acceleration=current_state.acceleration,  # Unpack from state
            final_value=new_target_value,
            duration=new_duration,
        )
        self._ctime = 0.0  # Reset clock for new transition
        self._value = float(new_target_value)  # Update target value
        self._duration = float(new_duration)  # Update duration

    @property
    def duration(self):
        """
        Get the duration of the transition.

        This is a unitless value. The actual time it takes depends on how `step()` is
        called.
        """
        return self._interpolator.duration if self._interpolator is not None else self._duration

    @property
    def value(self):
        """Get the smoothed value, according to the timing function."""
        if self._interpolator is None:
            return self._value

        if abs(self._ctime - self.duration) < 1e-10 or self._ctime > self.duration:
            return self._interpolator.final_value

        return self._interpolator(self._ctime)


class LogDynamicProp(DynamicProp):
    """A wrapper for DynamicProp that operates in logarithmic space."""

    def __init__(
        self,
        value: float,
        duration: float = 0.0,
        timing_function_cls: Type[TimingFunction] = MinimumJerkTimingFunction,
    ):
        """Initialize with a value in normal space and convert to log space internally."""
        # Ensure value is positive for log space
        safe_value = max(value, 1e-10)
        log_value = math.log(safe_value)

        # Initialize the parent SmoothProp in log space
        super().__init__(
            value=log_value,
            duration=duration,
            timing_function_cls=timing_function_cls,
        )

    def step(self, n=1.0):
        """Progress the internal clock forward."""
        super().step(n)

    def set(self, value: float | None = None, duration: float | None = None):
        """Set a new target value and/or duration for the transition."""
        # If value is provided, convert to log space
        if value is not None:
            # Ensure value is positive for log space
            safe_value = max(value, 1e-10)
            log_value = math.log(safe_value)
            super().set(value=log_value, duration=duration)
        else:
            super().set(value=None, duration=duration)

    @property
    def value(self):
        """Get the current value, transformed back from log space."""
        # Get the log-space value from the parent class
        log_value = super().value
        # Transform back to normal space
        return math.exp(log_value)

from typing import Type

from .dopesheet import Dopesheet
from .model import TStep, Frame
from .timing_fn import (
    LinearTimingFunction,
    MinimumJerkTimingFunction,
    StepEndTimingFunction,
    TimingFunction,
)
from .transitions import DynamicProp, LogDynamicProp

# Map interpolator names to classes
INTERPOLATOR_MAP: dict[str, Type[TimingFunction]] = {
    "minjerk": MinimumJerkTimingFunction,
    "linear": LinearTimingFunction,
    "step": StepEndTimingFunction,  # Allow 'step' as alias for 'step-end'
    "step-end": StepEndTimingFunction,
}
DEFAULT_TIMING_FUNCTION = MinimumJerkTimingFunction


class Timeline:
    """
    Evolves property values over time.

    Whereas the Dopesheet defines the properties and their keyframes,
    the Timeline is responsible for interpolating between those keyframes
    and updating the properties at each step.
    """

    props: dict[str, DynamicProp]
    _step: int
    _max_steps: int

    def __init__(self, dopesheet: Dopesheet):
        """Initialize the timeline."""
        self.dopesheet = dopesheet
        self._max_steps = len(self.dopesheet)

        # Get the initial values for each property from the dopesheet
        initial_values = dopesheet.get_initial_values()

        # Create SmoothProp instances with appropriate initial values and timing functions
        self.props = {}
        for prop in dopesheet.props:
            # Get the configuration for this property
            prop_config = dopesheet.get_prop_config(prop)

            # Look up the timing function class based on the config name
            timing_function_cls = INTERPOLATOR_MAP.get(prop_config.timing_fn, DEFAULT_TIMING_FUNCTION)

            # Use the initial value if available, otherwise default to 0.0
            initial_value = initial_values.get(prop, 0.0)

            # Create appropriate SmoothProp based on space setting
            if prop_config.space == "log":
                # Use LogSpaceSmoothProp for logarithmic space
                self.props[prop] = LogDynamicProp(
                    value=initial_value,
                    timing_function_cls=timing_function_cls,
                )
            else:
                # Use regular SmoothProp for linear space (the default)
                self.props[prop] = DynamicProp(
                    value=initial_value,
                    timing_function_cls=timing_function_cls,
                )

        self._step = 0
        # Set things in motion
        self._process_keyframes()

    def __len__(self) -> int:
        return self._max_steps

    def _process_keyframes(self) -> None:
        """Process keyframes at the current step."""
        current_step: Frame = self.dopesheet[self._step]

        for key in current_step.keyed_props:
            # Only proceed if we have a valid next target
            if key.next_t is None or key.next_value is None:
                continue

            prop_name = key.prop
            duration = key.duration

            # Set the target value and duration
            # The appropriate space transformation is handled by the SmoothProp or LogSpaceSmoothProp
            self.props[prop_name].set(value=key.next_value, duration=duration)

    def step(self) -> TStep:
        """Advance the timeline by one step."""
        if self._step >= self._max_steps:
            raise IndexError("Timeline has reached the end.")

        self._step += 1
        for prop in self.props.values():
            prop.step(1.0)
        self._process_keyframes()
        return self.state

    @property
    def state(self) -> TStep:
        """Get the current state of the timeline."""
        static_info = self.dopesheet[self._step]
        props = {prop: self.props[prop].value for prop in self.props}
        return TStep(
            step=self._step,
            phase=static_info.phase,
            actions=static_info.actions,
            props=props,
            is_phase_start=static_info.is_phase_start,
            is_phase_end=static_info.is_phase_end,
        )

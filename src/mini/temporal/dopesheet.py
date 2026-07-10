import logging
import re
from dataclasses import field
from io import BytesIO, StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Styler needs jinja2 (an optional dependency); only the styled=True path uses it.
    from pandas.io.formats.style import Styler

from .model import Keyframe, PropConfig, Frame

log = logging.getLogger(__name__)

RESERVED_COLS = ("STEP", "PHASE", "ACTION")


class Dopesheet:
    """
    A class to represent a dope sheet for parameter keyframes.

    ## Background
    A dope sheet (or exposure sheet) is a tool used in animation to organize and plan
    the timing of keyframes and actions. It typically helps animators visualize the
    sequence of events and manage the timing of actions effectively. It consists of a
    grid, where each row is a step in the animation, and each column represents a
    different property or action.

    ## Structure
    Dope sheets as defined by this class have the following columns:
    - STEP: The step/frame/epoch number
    - PHASE: The name of the phase of the curriculum (optional)
    - ACTION: The action to take (event to emit) (optional)
    - *: Other columns are interpreted as parameters to vary over time.

    https://en.m.wikipedia.org/wiki/Exposure_sheet
    """

    _df: pd.DataFrame
    _prop_configs: dict[str, PropConfig] = field(default_factory=dict)
    _phase_indices: np.ndarray

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Dopesheet with a DataFrame.

        Parses column headers for property configurations (e.g., 'x:log:minjerk')
        and resolves relative timesteps.

        See `from_csv`.
        """
        parsed_df, prop_configs = self._parse_header(df.copy())
        self._prop_configs = prop_configs
        self._df = resolve(parsed_df)
        self._phase_indices = self._df["PHASE"].dropna().index.to_numpy()

    @staticmethod
    def _parse_header(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, PropConfig]]:
        """Parse property configurations from DataFrame headers."""
        prop_configs = {}
        rename_map = {}

        for col in df.columns:
            if col in RESERVED_COLS:
                config = PropConfig(col)
            else:
                config = PropConfig.from_col_name(col)
            prop_configs[config.prop] = config
            if config.prop != col:
                rename_map[col] = config.prop

        if rename_map:
            df = df.rename(columns=rename_map)

        return df, prop_configs

    def __len__(self):
        """Get the number of steps in the dope sheet."""
        return self._df["STEP"].max() + 1

    def __getitem__(self, step: int) -> Frame:
        """
        Get the step details for the given step number.

        The sheet may not contain a keyframe for the given step. In that case, the
        current phase details will be returned without any keyed properties.
        """
        steps_col = self._df["STEP"]

        insertion_point = steps_col.searchsorted(step, side="right")
        idx = max(0, int(insertion_point - 1))

        phase_insertion_point = np.searchsorted(self._phase_indices, idx, side="right")

        if phase_insertion_point == 0:
            phase_idx = 0
        else:
            phase_idx = self._phase_indices[phase_insertion_point - 1]

        phase = str(self._df["PHASE"][phase_idx] or "")
        phase_start = bool(steps_col[phase_idx] == step)

        if phase_insertion_point < len(self._phase_indices):
            next_phase_df_idx = self._phase_indices[phase_insertion_point]
            next_phase_start_step = steps_col[next_phase_df_idx]
            current_phase_end_step = next_phase_start_step - 1
        else:
            current_phase_end_step = steps_col.max()

        is_phase_end = step == current_phase_end_step

        _t: int = steps_col[idx]
        if _t != step:
            return Frame(
                t=step,
                phase=phase,
                is_phase_start=False,
                is_phase_end=is_phase_end,
                actions=[],
                keyed_props=[],
            )

        action_value = self._df["ACTION"][idx]
        if pd.isna(action_value) or action_value == "":
            actions = []
        else:
            actions = str(action_value).split(",")

        keyed_props = []
        for prop in self.props:
            series = self._df[prop]
            value = series[idx]
            if pd.isna(value):
                continue
            next_idx = series[series.index > idx].first_valid_index()
            k = Keyframe(
                prop=prop,
                t=step,
                value=value,
                next_t=self._df["STEP"][next_idx] if next_idx is not None else None,
                next_value=series[next_idx] if next_idx is not None else None,
            )
            keyed_props.append(k)

        return Frame(
            t=step,
            phase=phase,
            is_phase_start=phase_start,
            is_phase_end=is_phase_end,
            actions=actions,
            keyed_props=keyed_props,
        )

    @property
    def props(self) -> list[str]:
        """List of all base property names in the dopesheet."""
        return [col for col in self._df.columns if col not in RESERVED_COLS]

    def get_prop_config(self, prop_name: str) -> PropConfig:
        """Get the parsed configuration for a specific property."""
        return self._prop_configs[prop_name]

    @property
    def phases(self) -> set[str]:
        """Return a set of unique phase names defined in the dopesheet."""
        return set(self._df["PHASE"].dropna().unique())

    def get_initial_values(self) -> dict[str, float]:
        """
        Get the initial value for each property in the dopesheet.

        For each property, finds the first non-NaN value in the timeline.
        If a property has no values, it will not be included in the result.

        Returns:
            A dictionary mapping property names to their initial values.
        """
        initial_values = {}
        for prop in self.props:
            series = self._df[prop]
            first_valid_idx = series.first_valid_index()
            if first_valid_idx is not None:
                initial_values[prop] = float(series[first_valid_idx])
        return initial_values

    @classmethod
    def from_csv(cls, path: Path | str | BytesIO | StringIO) -> "Dopesheet":
        """
        Load a dopesheet from a CSV file.

        The CSV file should have the following columns:
        - STEP: The step number (can be relative, e.g., +0.5)
        - PHASE: The phase of the curriculum (optional)
        - ACTION: The action to take (event to emit) (optional)
        - *: Other columns are interpreted as parameters to set.
             These can be in the following formats:
             - 'prop' (e.g., 'momentum') - Uses defaults for space and interpolator
             - 'prop:space' (e.g., 'lr:log') - Customizes space, uses default interpolator
             - 'prop::interpolator' (e.g., 'z::step-end') - Uses default space, customizes interpolator
             - 'prop:space:interpolator' (e.g., 'lr:log:minjerk') - Customizes both space and interpolator

             Default values: space='linear', interpolator='minjerk'

        Example:
            STEP,PHASE,ACTION,lr:log,momentum,z::step-end
            0,Basic,,0.01,0.9,1
            +0.5,,snapshot,0.005,,2
            1000,,,0.001,0.99,3
        """
        df = pd.read_csv(path, dtype={"STEP": str, "PHASE": str, "ACTION": str}, header=0)
        return cls(df)

    @overload
    def as_df(self, *, styled: Literal[True]) -> "Styler": ...
    @overload
    def as_df(self, *, styled: Literal[False] = False) -> pd.DataFrame: ...

    def as_df(self, *, styled=False) -> "pd.DataFrame | Styler":
        """Convert the dopesheet to a pandas DataFrame."""
        df = self._df.copy()
        if styled:
            df = style_dopesheet(df)
        return df

    def to_markdown(self) -> str:
        mdtable = self._df.to_markdown(index=False, tablefmt="pipe")
        mdtable = re.sub(r"(\|\s*)nan(\s*\|)", r"\1   \2", mdtable, flags=re.IGNORECASE)
        mdtable = re.sub(r"(\|\s*)nan(\s*\|)", r"\1   \2", mdtable, flags=re.IGNORECASE)
        return mdtable

    def to_dict(self):
        """
        Convert the dopesheet to a dictionary.

        Some logger frameworks like WandB will call this when storing model config.
        """
        df = self._df.copy()
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype(int)
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype(float)
            else:
                df[col] = df[col].astype(str).replace({"nan": None, "NaN": None})  # ty:ignore[invalid-argument-type]

        df = df.set_index("STEP", drop=False)
        df = df.rename(columns=lambda x: str(self.get_prop_config(cast(str, x))))
        return {
            col: series.dropna().to_dict()  #
            for col, series in df.items()
        }


def style_dopesheet(df: pd.DataFrame) -> "Styler":
    import pandas as pd

    decimal_places: dict[str, int] = {}
    non_numeric_cols: list[int] = []

    for i, col in enumerate(df.columns):
        if pd.api.types.is_numeric_dtype(df[col]):
            col_no_na = df[col].dropna()

            if col_no_na.empty:
                precision = 2
            else:
                is_integer = (col_no_na == col_no_na.round(0)).all()
                if is_integer:
                    precision = 0
                else:
                    precision = col_no_na.astype(str).str.split(".", expand=True)[1].str.len().max()
                    precision = int(precision) if pd.notna(precision) else 0
            decimal_places[col] = min(precision, 6)
        elif col == "STEP":
            pass
        else:
            non_numeric_cols.append(i)

    log.info(f"Calculated decimal places: {decimal_places}")
    log.info(f"Non-numeric columns: {non_numeric_cols}")

    style = df.style.set_table_styles(
        [
            {"selector": "td,th", "props": "white-space: nowrap"},
            *[{"selector": f".col{i}", "props": "text-align: left"} for i in non_numeric_cols],
        ]  # ty:ignore[invalid-argument-type]
    ).format(na_rep="")
    for i, precision in decimal_places.items():
        style = style.format(na_rep="", precision=precision, subset=[i])

    return style


def resolve(df: pd.DataFrame) -> pd.DataFrame:
    df["STEP"] = resolve_timesteps(df["STEP"])
    df = df.sort_values(by="STEP", ignore_index=True).reset_index(drop=True)
    return df


def _identify_anchors(steps: pd.Series) -> tuple[pd.Index, dict[int, int], pd.Series]:
    """Identify anchor steps (non-negative integers) and initialize resolved series."""
    resolved_steps = pd.Series(pd.NA, index=steps.index, dtype="Int64")
    anchor_indices_list = []
    anchor_steps_dict: dict[int, int] = {}

    for idx, step_str in steps.items():
        if not step_str.startswith(("-", "+")):
            try:
                if not re.fullmatch(r"\d+", step_str):
                    log.warning(
                        f"Warning: Absolute step '{step_str}' at index {idx} is not a valid non-negative integer. Treating as invalid."
                    )
                    continue

                step_val = int(step_str)
                anchor_steps_dict[cast(int, idx)] = step_val
                resolved_steps.loc[idx] = step_val  # type: ignore
                anchor_indices_list.append(idx)
            except ValueError:
                log.warning(f"Warning: Could not parse absolute step '{step_str}' at index {idx} as integer.")

    anchor_indices = pd.Index(anchor_indices_list)
    return anchor_indices, anchor_steps_dict, resolved_steps


def _resolve_integer_offset(
    prefix: str,
    offset_int: int,
    idx: int,
    step_str: str,
    prev_step: int | None,
    next_step: int | None,
) -> int | None:
    """Resolve +N or -N relative steps."""
    resolved_step: int | None = None
    if prefix == "+":
        if prev_step is not None:
            resolved_step = prev_step + offset_int
        else:
            log.warning(f"Warning: Cannot resolve relative step '{step_str}' at index {idx}: No preceding anchor.")
    elif prefix == "-":
        if next_step is not None:
            resolved_step = next_step - offset_int
        else:
            log.warning(f"Warning: Cannot resolve relative step '{step_str}' at index {idx}: No succeeding anchor.")
    return resolved_step


def _resolve_fractional_offset(
    prefix: str,
    fraction: float,
    idx: int,
    step_str: str,
    prev_step: int | None,
    next_step: int | None,
) -> int | None:
    """Resolve +F or -F relative steps."""
    if prev_step is None or next_step is None:
        log.warning(f"Warning: Cannot resolve fractional step '{step_str}' at index {idx}: Missing bracketing anchors.")
        return None

    interval = next_step - prev_step
    resolved_step: int | None = None
    if prefix == "+":
        resolved_step = round(prev_step + fraction * interval)
    elif prefix == "-":
        resolved_step = round(next_step - fraction * interval)
    return resolved_step


def _resolve_single_relative(
    step_str: str,
    idx: int,
    prev_anchor_idx: int,
    next_anchor_idx: int,
    anchor_steps: dict[int, int],
) -> int | None:
    """Resolve a single relative step string (+N, -N, +F, -F) by dispatching."""
    prefix = step_str[0]
    value_str = step_str[1:]
    prev_step = anchor_steps.get(prev_anchor_idx)
    next_step = anchor_steps.get(next_anchor_idx)

    if re.fullmatch(r"[1-9]\d*", value_str):
        try:
            offset_int = int(value_str)
            return _resolve_integer_offset(prefix, offset_int, idx, step_str, prev_step, next_step)
        except ValueError:
            pass

    try:
        if not re.fullmatch(r"\d*\.\d+", value_str) and not re.fullmatch(r"\.\d+", value_str):
            raise ValueError("Not a float format for fractional step (missing decimal?).")

        fraction = float(value_str)
        if not (0 < fraction < 1):
            log.warning(
                f"Warning: Fractional step '{step_str}' at index {idx} must have value strictly between 0 and 1."
            )
            return None

        return _resolve_fractional_offset(prefix, fraction, idx, step_str, prev_step, next_step)

    except ValueError:
        log.warning(
            f"Warning: Could not parse value from relative step '{step_str}' at index {idx} as positive integer or 0<float<1."
        )
        return None


def resolve_timesteps(steps: pd.Series) -> pd.Series:
    """
    Resolve absolute and relative timesteps in a dopesheet STEP column.

    Handles:
    - Absolute steps: Non-negative integers (e.g., '0', '10').
    - Relative integer steps: '+N' (N steps after previous anchor),
                              '-N' (N steps before next anchor). N must be positive.
    - Relative fractional steps: '+F' (interpolate between prev/next anchors),
                                 '-F' (interpolate backwards from next anchor). 0 < F < 1.
    - Invalid formats result in pd.NA.
    - Resolved negative steps are clamped to 0.
    """
    steps = steps.astype(str)
    anchor_indices, anchor_steps_dict, resolved_steps = _identify_anchors(steps)

    relative_indices = steps.index.difference(anchor_indices)

    for idx in relative_indices:
        step_str = steps.loc[idx]

        if not step_str.startswith(("-", "+")):
            log.warning(
                f"Warning: Step '{step_str}' at index {idx} is neither anchor nor relative. Treating as invalid."
            )
            continue

        prev_anchor_idx = max((a_idx for a_idx in anchor_indices if a_idx < idx), default=-1)
        next_anchor_idx = min((a_idx for a_idx in anchor_indices if a_idx > idx), default=-1)

        resolved_val = _resolve_single_relative(step_str, idx, prev_anchor_idx, next_anchor_idx, anchor_steps_dict)

        if resolved_val is not None:
            if resolved_val < 0:
                log.warning(
                    f"Warning: Resolved step for '{step_str}' at index {idx} is negative ({resolved_val}). Clamping to 0."
                )
                resolved_val = 0
            resolved_steps.loc[idx] = resolved_val

    return resolved_steps.astype("Int64")

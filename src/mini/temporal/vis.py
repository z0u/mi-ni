import re
from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from pandas.api.types import is_numeric_dtype

from mini.vis import light_dark
from .dopesheet import RESERVED_COLS
from .timeline import Timeline


def realize_timeline(timeline: Timeline) -> pd.DataFrame:
    # Collect data by stepping through the timeline
    history = []
    max_steps = len(timeline.dopesheet)
    for _ in range(max_steps + 1):
        state = timeline.state
        history.append({"STEP": state.step, "PHASE": state.phase, "ACTION": state.actions, **state.props})
        if state.step < max_steps:
            timeline.step()

    return pd.DataFrame(history)


@dataclass
class ParamGroup:
    """A group of parameters to be plotted together."""

    name: str
    params: list[str]
    height_ratio: float = 1.0
    """Height ratio for the subplot."""


def group_properties_by_scale(df: pd.DataFrame) -> tuple[ParamGroup, ParamGroup]:
    """Group properties by their scale for better visualization."""
    # Filter for numeric columns first!
    numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
    if not numeric_cols:
        # Handle case where there are no numeric columns
        return ParamGroup(name="", params=[], height_ratio=2.0), ParamGroup(name="", params=[], height_ratio=1.0)

    # Calculate statistics only for numeric properties
    prop_stats = {
        prop: {"max": df[prop].max(), "median": df[prop].median(), "min": df[prop].min()}
        for prop in numeric_cols  # Use the filtered list
    }

    # Calculate median of medians using only numeric stats
    median_values = [
        stats["median"] for stats in prop_stats.values() if pd.notna(stats["median"])
    ]  # Ensure median is not NaN
    if not median_values:
        # Handle case where all medians are NaN (e.g., all columns are empty or NaN)
        median_of_medians = 0
    else:
        median_of_medians = np.median(median_values)

    # Group properties - those with much smaller medians go to group 2
    # Handle cases where median_of_medians might be zero or very small
    threshold = median_of_medians * 0.4 if median_of_medians > 1e-9 else 1e-9
    group1 = [prop for prop in numeric_cols if prop_stats[prop]["max"] >= threshold]
    group2 = [prop for prop in numeric_cols if prop_stats[prop]["max"] < threshold]

    return ParamGroup(name="", params=group1, height_ratio=2.0), ParamGroup(name="", params=group2, height_ratio=1.0)


def get_styles(
    prop: str, line_styles: list[tuple[re.Pattern, dict[str, Any]]], default_style: dict[str, Any] | None = None
):
    """Get line styles for plotting."""
    # Convert str matchers to regex patterns, escaping special characters
    for pattern, style in line_styles:
        if pattern.search(prop):
            line_style = style
            break
    else:
        # Default line style if no pattern matches
        line_style = {}
    return {**(default_style or {}), **line_style}


def plot_timeline(  # noqa: C901
    history_df: pd.DataFrame,
    keyframes_df: pd.DataFrame,
    groups: Sequence[ParamGroup] | None = None,
    ax: Axes | None = None,
    show_legend: bool = True,
    legend_fontsize: str | None = None,
    title: str = "Timeline property evolution",
    show_phase_labels: bool = True,
    line_styles: Sequence[tuple[str | re.Pattern, dict[str, Any]]] | None = None,
):
    if groups is None:
        cols = [col for col in history_df.columns if col not in RESERVED_COLS]
        groups = [ParamGroup(name="", params=cols, height_ratio=1.0)]

    # Convert str matchers to regex patterns, escaping special characters
    compiled_styles: list[tuple[re.Pattern[str], dict[str, Any]]] = [
        (re.compile(re.escape(pattern)) if isinstance(pattern, str) else pattern, style)
        for pattern, style in line_styles or []
    ]

    # --- Figure/Axes Setup ---
    if ax is None:
        # If no axis provided, create a new figure and axes based on groups
        height_ratios = [g.height_ratio for g in groups]
        figsize = (15, 3.5 * len(groups))
        fig, axes_list = plt.subplots(
            len(groups),
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
            squeeze=False,  # Always return a 2D array
            constrained_layout=True,
        )
        # Use the first axes from the created list if multiple groups, else the single axes
        main_ax = axes_list[0, 0]
        axes_to_plot_on = axes_list.flatten()
    else:
        # If an axes is provided, use it. Assume only one group can be plotted.
        if len(groups) > 1:
            print("Warning: Multiple groups provided but only one axes given. Plotting only the first group.")
        fig = ax.get_figure()
        main_ax = ax  # The main axes is the one provided
        axes_to_plot_on = [ax]  # Plot only on the provided axes
        groups = [groups[0]]  # Use only the first group

    marker_color = light_dark("#444", "#aaa")

    # --- Plotting Data ---
    # Plot each group on its corresponding axes (or the single provided axes)
    for group, current_ax in zip(groups, axes_to_plot_on, strict=True):
        # Initialize legend items for the current axes
        lines_for_current_ax: list[Artist] = []
        labels_for_current_ax: list[str] = []

        if current_ax.get_figure() is None:  # Check if axes belongs to a figure
            raise ValueError("Provided axes does not belong to a figure.")
        for prop in group.params:
            # Ensure the property exists in the history dataframe before plotting
            if prop in history_df.columns:
                line_style = get_styles(prop, compiled_styles, {"alpha": 0.75})
                (line,) = current_ax.plot(history_df["STEP"], history_df[prop], label=f"{prop}", **line_style)
                # Add to the current axes' legend items
                lines_for_current_ax.append(line)
                labels_for_current_ax.append(f"{prop}")

                # Add markers for keyframes if the property exists in keyframes
                if prop in keyframes_df.columns:
                    prop_keyframes = keyframes_df.dropna(subset=[prop])
                    if not prop_keyframes.empty:
                        current_ax.scatter(
                            prop_keyframes["STEP"],
                            prop_keyframes[prop],
                            marker="o",
                            facecolor=light_dark("white", "black"),
                            s=25,
                            zorder=5,
                            color=line.get_color(),
                        )
            else:
                print(f"Warning: Property '{prop}' specified in group '{group.name}' not found in history_df.")

        # --- Legend (Create legend for the current axes if show_legend is True) ---
        if show_legend and lines_for_current_ax:
            # Special handling for the main_ax to potentially include 'Action Triggered'
            if current_ax == main_ax:
                # Check if actions exist and add the handle if needed
                action_steps = history_df[history_df["ACTION"].apply(lambda x: bool(x))]
                if not action_steps.empty:
                    action_handle = Line2D(
                        [0],
                        [0],
                        marker="^",
                        color=marker_color,
                        linestyle="None",  # No line for marker-only legend entry
                        markersize=8,  # Adjust size if needed
                        label="Action Triggered",
                    )
                    # Avoid duplicates if somehow already added (shouldn't happen now)
                    if "Action Triggered" not in labels_for_current_ax:
                        lines_for_current_ax.append(action_handle)
                        labels_for_current_ax.append("Action Triggered")

            # Create the legend for the current axes
            by_label = dict(zip(labels_for_current_ax, lines_for_current_ax, strict=True))
            current_ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=legend_fontsize)

    # --- Phase Changes and Labels (Plot only on main_ax) ---
    phase_changes = keyframes_df.dropna(subset=["PHASE"])
    last_phase = None
    phase_boundaries = []
    for _, row in phase_changes.iterrows():
        if row["PHASE"] != last_phase:
            step = row["STEP"]
            # Draw vertical line only on the main axes
            main_ax.axvline(step, color="grey", alpha=0.2)
            phase_boundaries.append({"STEP": step, "PHASE": row["PHASE"]})
            last_phase = row["PHASE"]

    # Add phase labels only to the main plot (main_ax)
    if show_phase_labels:
        for i, pb in enumerate(phase_boundaries):
            mid_point = (
                (phase_boundaries[i + 1]["STEP"] + pb["STEP"]) / 2
                if i + 1 < len(phase_boundaries)
                else (history_df["STEP"].max() + pb["STEP"]) / 2  # Use max step from history
            )
            main_ax.text(
                mid_point,
                main_ax.get_ylim()[1],  # Position at the top
                pb["PHASE"],
                ha="center",
                va="bottom",  # Align bottom of text to top of plot
                fontweight="bold",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc=light_dark("none", "#222"), alpha=0.7, ec="none"),
            )

    # --- Action Markers (Plot only on main_ax) ---
    action_steps = history_df[history_df["ACTION"].apply(lambda x: bool(x))]  # Filter steps with non-empty action lists
    if not action_steps.empty:
        y_min, _y_max = main_ax.get_ylim()
        marker_y_pos = y_min
        main_ax.scatter(
            action_steps["STEP"].unique(),
            [marker_y_pos] * len(action_steps["STEP"].unique()),
            marker="^",
            color=marker_color,
            s=40,
            zorder=10,
        )

    # --- Labels, Title, Grid (Apply to main_ax) ---
    if ax is None:  # Only set title if we created the figure
        main_ax.set_title(title)
    main_ax.set_ylabel("Property value")
    main_ax.set_xlabel("Step")

    # Return the figure and the main axes (or list of axes if created internally)
    return fig, main_ax if ax is not None else axes_to_plot_on

# title: Themed plots

"""
# Themed plots

`themed` wraps a plot function to render in both light and dark modes, producing a single HTML element that switches on `prefers-color-scheme`. The same function runs twice — once per theme — so you can use `light_dark()` inside to pick theme-dependent values.

This page is the `docs/themed.py` notebook, rewritten as a literate document: a plain Python file where each `# %%` starts a cell and each top-level string is prose.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mini.lit import memo
from mini.vis import themed
from mini.vis.theme import light_dark

x = np.linspace(0, 2 * np.pi, 300)

"""
## Plain decorator

The simplest form: `@themed` with no arguments. A cell's last expression is displayed, and `themed` returns the figure's HTML, so the figure appears here.
"""


# %%
@themed
def plot_plain() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, np.sin(x), color=light_dark("#1a5f8a", "#6ab0d4"), lw=2)
    ax.set_title("sin(x)")
    return fig


plot_plain()

"""
## Decorator factory

Pass keyword arguments to set `alt_text`, `caption`, `max_width`, or custom styles. This is the form you want when defining a standalone plot function. Stacking `@memo` outside it caches the rendered HTML (and keeps the two PNGs it wrote) keyed by the function's source and arguments, so re-rendering this page after a prose edit does not redraw the figure.
"""


# %%
@memo
@themed(alt_text="sin and cos", caption="Two sinusoids, a quarter period apart.")
def plot_factory(x: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    color_sin = light_dark("#1a5f8a", "#6ab0d4")
    color_cos = light_dark("#8a3a1a", "#d49a6a")
    ax.plot(x, np.sin(x), color=color_sin, lw=2, label="sin")
    ax.plot(x, np.cos(x), color=color_cos, lw=2, label="cos")
    ax.legend()
    return fig


plot_factory(x)

"""
## Direct call

Useful for one-off plots, or when wrapping a function defined elsewhere. Here the cell's code is hidden (`# %% hide`), which is how a report shows a figure without its plumbing.
"""


# %% hide
def _plot_raw() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, np.sin(x) * np.exp(-x / 6), color=light_dark("#2a6e3a", "#7ad49a"), lw=2)
    ax.set_title("Damped sine")
    return fig


themed(_plot_raw, alt_text="Damped sine wave")()

"""
The prose can also quote values from the namespace: the grid has {{ x.size }} points and its last value is {{ "%.3f"|format(x[-1]) }}.
"""

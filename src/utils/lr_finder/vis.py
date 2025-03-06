from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

from utils.lr_finder.types import LRFinderConfig, LRFinderSeries
from utils.param_types import validate_call
from utils.theming import fig_theme_toggle


@validate_call
def lr_finder_plot() -> Callable[[LRFinderSeries | LRFinderConfig], None]:
    from IPython.display import HTML, display, DisplayHandle

    history: list[LRFinderSeries] = []
    disp = cast(DisplayHandle, display(display_id=True))
    # default config
    config = LRFinderConfig()

    def handle(event: LRFinderSeries | LRFinderConfig):
        """Receive and process a new series."""
        nonlocal history, config
        if isinstance(event, LRFinderConfig):
            config = event
        elif isinstance(event, LRFinderSeries):
            history.append(event)
            with plt.ioff(), plt.style.context('dark_background'):
                fig = _draw(history, config)
                disp.update(HTML(fig_theme_toggle(fig, already_dark=True)))
                plt.close(fig)

    return handle


def _draw(history: list[LRFinderSeries], config: LRFinderConfig) -> Figure:
    """Update the visualization with new data."""
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    fig.suptitle(f'Learning Rate Finder (Multi-Scale Search) ({config.method.title()})', y=0.95)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_xscale('log')

    # Create color map for zoom history
    cmap = plt.get_cmap('viridis')
    norm = Normalize(0, config.num_zooms)

    # Plot historical scales
    for h1, h2 in zip(history[1:], history[:-1], strict=True):
        color = cmap(norm(h1.zoom))

        # Create fill between consecutive curves
        xs = np.concatenate([h1.lrs, h2.lrs[::-1]])
        ys = np.concatenate([h1.losses, h2.losses[::-1]])
        ax.fill(xs, ys, color=color, alpha=0.3)
        ax.semilogx(h1.lrs, h1.losses, color=color, linewidth=1)

    # Plot latest/final scale
    series = history[-1]
    color = cmap(norm(series.zoom))
    ax.semilogx(series.lrs, series.losses, color=color, linewidth=1)
    ax.axvline(x=series.best_lr, color='white', linestyle='--', label='Suggested LR')

    # Plot progression
    best_lrs = []
    best_losses = []
    for series in history:
        loss = np.exp(np.interp(np.log(series.steepest_lr), np.log(series.lrs), np.log(series.losses)))
        best_lrs.append(series.steepest_lr)
        best_losses.append(loss)

    ax.semilogx(
        best_lrs,
        best_losses,
        '-',
        label='Steepest gradient (weighted av.)',
        color='white',
        linewidth=1,
        markerfacecolor='black',
        markeredgecolor='white',
        markeredgewidth=1,
        markersize=4,
    )

    ax.legend(loc='upper left')
    fig.tight_layout()
    return fig

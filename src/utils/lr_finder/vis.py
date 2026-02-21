import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from mini.vis.plt import Theme
from utils.lr_finder.types import LRFinderConfig, LRFinderSeries


def plot_lr_finder(
    history: list[LRFinderSeries],
    config: LRFinderConfig,
    *,
    theme: Theme | None = None,
) -> Figure:
    """Plot multi-scale LR finder search results."""
    theme = theme or Theme('indeterminate')
    fig, ax = plt.subplots(figsize=(10, 3))

    fig.suptitle(f'Learning Rate Finder ({config.method.title()})', y=0.95)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_xscale('log')

    cmap = plt.get_cmap('viridis')
    norm = Normalize(0, config.num_zooms)

    # Fill between consecutive zoom curves
    for h1, h2 in zip(history[1:], history[:-1], strict=True):
        color = cmap(norm(h1.zoom))
        xs = np.concatenate([h1.lrs, h2.lrs[::-1]])
        ys = np.concatenate([h1.losses, h2.losses[::-1]])
        ax.fill(xs, ys, color=color, alpha=0.3)
        ax.semilogx(h1.lrs, h1.losses, color=color, linewidth=1)

    # Final zoom scale
    series = history[-1]
    color = cmap(norm(series.zoom))
    ax.semilogx(series.lrs, series.losses, color=color, linewidth=1)
    ax.axvline(
        x=series.best_lr,
        color=theme.val('#666', light='#666', dark='#aaa'),
        linestyle='--',
        label='Suggested LR',
    )

    # Steepest-gradient progression across zooms
    best_lrs = []
    best_losses = []
    for s in history:
        loss = np.exp(np.interp(np.log(s.steepest_lr), np.log(s.lrs), np.log(s.losses)))
        best_lrs.append(s.steepest_lr)
        best_losses.append(loss)

    ax.semilogx(
        best_lrs,
        best_losses,
        '-',
        label='Steepest gradient (weighted av.)',
        color=theme.val('#111', light='#111', dark='#eee'),
        linewidth=1,
        markerfacecolor=theme.val('#fff', light='#fff', dark='#111'),
        markeredgecolor=theme.val('#111', light='#111', dark='#eee'),
        markeredgewidth=1,
        markersize=4,
    )

    ax.legend(loc='upper left')
    fig.tight_layout()
    return fig

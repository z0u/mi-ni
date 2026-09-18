`mini.vis` interface:

```py
def themed(plot: Callable[P, Figure]) -> Callable[P, str]:
    """Wrap a plot function to apply styles. Returns HTML."""

def light_dark[T](light: T, dark: T) -> T:
    """Pick a value based on the active theme (like CSS ``light-dark()``)."""
```

`themed` wraps a plot function to render in both light and dark modes, producing a single HTML element that switches on `prefers-color-scheme`. The same function runs twice — once per theme — so you can use `light_dark()` inside to pick theme-dependent values. It can be used as a decorator with or without arguments:

```py
@themed(alt_text="Plot of a sine wave")
def plot_factory() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, np.sin(x), color=light_dark("#1a5f8a", "#6ab0d4"), lw=2)
    ax.set_title("sin(x)")
    return fig

plot_factory()
```

## Externalizing figures (reports)

By default a `themed` figure inlines as a `data:` URI — fine for one figure, heavy for a report (two PNGs each, light + dark), and past a certain weight it makes for an unwieldy page. Set a _publisher_ once near the top of the report and `themed` writes each figure out to a file (keyed by its readable name) referenced by a relative URL instead. Plot code doesn't change:

```py
from mini.vis import themed
from mini.reports import use_publisher, report_bundle

use_publisher(report_bundle(__file__))   # → _assets/ when exporting, a scratch dir while serving

@themed(alt_text="…", name="loss-curve")  # name → loss-curve-{light,dark}.png
def _plot(): ...
_plot()
```

`name` (default: the plot function's name) is the figure's readable basename — it ends up in the asset filename and the saved-file name, and on a `data-asset-name` attribute for provenance. The publisher, the `asset_url` verb for arbitrary data blobs, and how the bundle reaches the web (the `<base>` switch + the relative-links rule) all live in [reports.md](./reports.md).

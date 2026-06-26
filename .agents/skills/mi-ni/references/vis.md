`mini.vis` interface:

```py
def themed(plot: Callable[P, Figure]) -> Callable[P, str]:
    """Wrap a plot function to apply styles. Returns HTML."""

def light_dark[T](light: T, dark: T) -> T:
    """Pick a value based on the active theme (like CSS ``light-dark()``)."""
```

`themed` wraps a plot function to render in both light and dark modes, producing
a single HTML element that switches on `prefers-color-scheme`. The same function
runs twice — once per theme — so you can use `light_dark()` inside to pick
theme-dependent values. It can be used as a decorator with or without arguments:

```py
@themed(alt_text='Plot of a sine wave')
def plot_factory() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, np.sin(x), color=light_dark('#1a5f8a', '#6ab0d4'), lw=2)
    ax.set_title('sin(x)')
    return fig

mo.Html(plot_factory())
```

## Externalizing figures and assets (reports)

By default a `themed` figure inlines as a `data:` URI — fine to view, heavy for a
report (two PNGs per figure, light + dark). A `Publisher` instead writes each asset
out as a content-addressed file and references it by a **relative** URL, keeping the
report HTML light. Set one up once in the report's setup cell; figure cells don't
change:

```py
from mini.vis import themed, use_publisher, report_bundle

use_publisher(report_bundle(__file__))   # assets → this report's __marimo__/_assets/

@themed(alt_text='…')
def _plot(): ...
mo.Html(_plot())
```

`report_bundle(__file__)` points assets at `…/__marimo__/_assets/` beside the
exported HTML, so the relative `_assets/<sha>.png` URL resolves there. With no
publisher, figures inline as before (a self-contained export still works). For
arbitrary blobs a report's JS reads (a large JSON a data browser fetches, an SPA's
data files), use the publisher directly — `asset_url` writes the bytes and returns
the same kind of relative URL:

```py
pub = use_publisher(report_bundle(__file__))
url = pub.asset_url(points_json, name='points.json')   # -> '_assets/<sha>.json'
```

**How it reaches the web.** The relative URL is the point: the *same* HTML works two
ways. Opened locally, `_assets/…` resolves to the co-located files (offline; the
figures are real PNGs you can open). Published, `scripts/build_site.py` uploads
`_assets/` to the HF bucket and inserts one `<base href>` in the `<head>` so the same
relative URLs resolve there — no per-URL rewriting. Because `<base>` repoints *every*
relative URL, the rule is **the only relative URLs in a report are its assets**; make
nav/source links absolute (e.g. to GitHub source), and `build_site` warns on stray
ones. Design notes: `research/reports-hfstore-migration.md`.

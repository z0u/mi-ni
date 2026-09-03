"""
Notebook utilities for rendering themed matplotlib figures as HTML.

A report's figures are heavy (a themed plot is *two* PNGs, light and dark). Inlined as ``data:`` URIs they bloat the exported HTML — the bytes Git LFS used to carry — and a page of them exceeds the output size marimo will display, so a heavy report won't render interactively either. A :class:`~mini.reports.Publisher` instead writes each blob out as a file (keyed by its readable name) and references it by a **relative** URL, so the HTML stays light. Set one up once per report and every ``@themed`` figure externalizes with no per-figure ceremony::

    # in the report's setup cell
    from mini.vis import themed
    from mini.reports import use_publisher, report_bundle
    use_publisher(report_bundle(__file__))

    # in a figure cell — unchanged
    @themed(alt_text='…')
    def _plot(): ...
    mo.Html(_plot())

The relative reference is the point: the *same* HTML works every way it's read. Opened locally it resolves to the co-located ``_assets/`` files (offline, and the figures are real PNG files); published, ``scripts/build_site.py`` uploads those files to the HF bucket and inserts a single ``<base href>`` so the very same relative URLs resolve there. Interactively, :func:`~mini.reports.report_bundle` points the publisher at the notebook's ``public/.mini/`` instead, the one place marimo's dev server serves files from. A figure with no publisher at all inlines as a self-contained ``data:`` URI. The publisher and the bundle protocol live in :mod:`mini.reports`.
"""

from __future__ import annotations

import logging
from functools import wraps
from textwrap import dedent
from typing import Callable, ParamSpec, TypeVar, overload

from .plt import use_style
from .theme import use_theme

from collections.abc import Sequence

from matplotlib.figure import Figure

from mini.reports import Publisher, current_publisher
from mini.vis.plt import Stylesheet


__all__ = ["figure_html", "themed", "themed_figure_html"]

P = ParamSpec("P")
R = TypeVar("R")


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Themed figures
# ---------------------------------------------------------------------------


@overload
def themed(
    plot: Callable[P, Figure],
    *,
    alt_text: str | None = ...,
    caption: str | None = ...,
    max_width: str | None = ...,
    name: str | None = ...,
    publish: Publisher | None = ...,
    light_styles: Sequence[Stylesheet] = ...,
    dark_styles: Sequence[Stylesheet] = ...,
) -> Callable[P, str]: ...


@overload
def themed(
    plot: None = ...,
    *,
    alt_text: str | None = ...,
    caption: str | None = ...,
    max_width: str | None = ...,
    name: str | None = ...,
    publish: Publisher | None = ...,
    light_styles: Sequence[Stylesheet] = ...,
    dark_styles: Sequence[Stylesheet] = ...,
) -> Callable[[Callable[P, Figure]], Callable[P, str]]: ...


def themed(
    plot: Callable[P, Figure] | None = None,
    *,
    alt_text: str | None = None,
    caption: str | None = None,
    max_width: str | None = None,
    name: str | None = None,
    publish: Publisher | None = None,
    light_styles: Sequence[Stylesheet] = ("base", "light"),
    dark_styles: Sequence[Stylesheet] = ("base", "dark"),
) -> Callable[P, str] | Callable[[Callable[P, Figure]], Callable[P, str]]:
    """Wrap a plot function to render in both light and dark themes.

    Inside each call, :func:`~mini.vis.plt.use_theme` sets an active theme so the plot can use :func:`~mini.vis.plt.light_dark` to pick theme-dependent values.

    Can be used as a plain decorator, a decorator factory, or called directly::

        @themed
        def plot(): ...

        @themed(alt_text='My plot')
        def plot(): ...

        themed(plot_lr_finder, alt_text='LR finder')(lr_history, lr_config)

    By default the figure is inlined as a ``data:`` URI. To externalize it (keeping the report HTML light enough to render and to ship), set a default :class:`~mini.reports.Publisher` with :func:`~mini.reports.use_publisher`, or pass ``publish=`` one here. *name* is the externalized figure's readable basename (it ends up in the asset filename and the download name); it defaults to the plot function's name.

    *caption* is **Markdown** rendered into a ``<figcaption>`` inside the ``<figure>``, so the caption travels with the image instead of riding along in a sibling ``mo.vstack``. Write it as a triple-quoted string — ``mo.md`` dedents, so leading indentation is stripped for you.

    The plot function is called once per theme, so everything inside it runs twice — including work that has nothing to do with color. Compute the data in the cell, outside the decorated function, and let it receive and draw. This matters most for anything sampled or fitted: a null distribution, a bootstrap, a bisection.
    """

    def decorator(fn: Callable[P, Figure]) -> Callable[P, str]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
            with use_theme("light"), use_style(*light_styles):
                light_fig = fn(*args, **kwargs)
            with use_theme("dark"), use_style(*dark_styles):
                dark_fig = fn(*args, **kwargs)

            if light_fig is None or dark_fig is None:
                msg = f"{fn.__name__} returned None"
                raise ValueError(msg)

            return themed_figure_html(
                light_fig,
                dark_fig,
                alt_text=alt_text,
                caption=_render_caption(caption),
                max_width=max_width,
                name=name or getattr(fn, "__name__", "").lstrip("_") or "figure",
                publish=publish if publish is not None else current_publisher(),
            )

        return wrapper

    if plot is not None:
        return decorator(plot)
    return decorator


def _render_caption(caption: str | None) -> str | None:
    """Render a Markdown caption to an HTML fragment for a ``<figcaption>``.

    Marimo owns the Markdown pipeline, so we defer to :func:`marimo.md` (and import it lazily — the library core stays marimo-free for callers that never caption). It dedents internally, so a triple-quoted string with indentation renders cleanly.
    """
    if caption is None:
        return None
    import marimo as mo

    return mo.md(caption).text


def figure_html(
    body: str,
    *,
    caption: str | None = None,
    aria_label: str | None = None,
    class_: str | None = None,
    style: str | None = None,
) -> str:
    """Wrap an HTML/SVG *body* in a ``<figure>``, optionally with a ``<figcaption>``.

    The shared seam behind themed figures, subline strips, and captioned tables: it only assembles the element, staying agnostic about how *body* and *caption* were produced and how they're styled (that is left to CSS or the caller). *caption* is an HTML fragment; render Markdown with :func:`marimo.md` first if you have it.

    *aria_label* gives the figure an accessible name for when the body is a group of marks that reads as one picture with no text of its own — e.g. a strip of inline SVGs. It is a plain ``aria-label`` (not ``role="img"``): a figure takes its name from the label without becoming atomic, so any sub-figures and their captions stay navigable. (``role="img"`` would make the subtree presentational and hide them — the reason to avoid it for a captioned group.)
    """
    import html

    attrs = ""
    if class_ is not None:
        attrs += f' class="{html.escape(class_)}"'
    if style is not None:
        attrs += f' style="{html.escape(style)}"'
    if aria_label is not None:
        # Collapse whitespace so a triple-quoted label reads as one line in the export.
        attrs += f' aria-label="{html.escape(" ".join(aria_label.split()))}"'
    figcaption = f"<figcaption>{caption}</figcaption>" if caption is not None else ""
    return f"<figure{attrs}>{body}{figcaption}</figure>"


def themed_figure_html(
    light_fig: Figure,
    dark_fig: Figure,
    *,
    close_fig: bool = True,
    alt_text: str | None = None,
    caption: str | None = None,
    max_width: str | None = None,
    name: str | None = None,
    publish: Publisher | None = None,
    **savefig_kwargs: str | int | bool,
) -> str:
    """Render light/dark matplotlib figures as an HTML figure element.

    With ``publish`` set, each PNG is written out and referenced by a relative URL (named ``<name>-light.png`` / ``<name>-dark.png`` so a saved file reads sensibly); otherwise both inline as ``data:`` URIs. *name* is also surfaced as a ``data-asset-name`` attribute on each ``<img>`` for provenance.

    *caption* is an HTML fragment (not Markdown — this seam is renderer-agnostic) placed in a ``<figcaption>`` inside the ``<figure>``. The :func:`themed` decorator renders its Markdown ``caption`` here for you.

    Each ``<img>`` carries explicit ``width``/``height`` attributes: the figure's *physical* size (PNG pixels × 96 CSS px/in ÷ save dpi), not its pixel count. Without them the browser displays 1 image px per CSS px, so the render dpi would leak into layout — a 192 dpi figure would paint at twice its figsize, and text sized to match the page would not. Pinning the CSS size makes the extra pixels crispness on high-dpr screens instead of extra inches, and lets the browser reserve the right space before the image loads.
    """
    import base64
    import hashlib
    import html
    from io import BytesIO

    import matplotlib.pyplot as plt

    defaults = {
        "bbox_inches": "tight",
        # 96×2: an exact 2:1 device-pixel ratio on retina/high-dpr screens, so the
        # extra pixels land as crispness rather than a fractional upscale (blurry at
        # 150, which is only 1.5625× the 96 px/in CSS reference).
        "dpi": 192,
    }
    save_args = defaults | savefig_kwargs

    def _png_bytes(fig: Figure) -> bytes:
        img_io = BytesIO()
        fig.savefig(img_io, format="png", facecolor=fig.get_facecolor(), **save_args)  # ty:ignore[invalid-argument-type]
        return img_io.getvalue()

    light_png = _png_bytes(light_fig)
    dark_png = _png_bytes(dark_fig)

    def _css_size(png: bytes) -> tuple[int, int]:
        # PNG pixel dims from the IHDR chunk (fixed offset in every PNG), scaled to
        # CSS px at the reference 96 px/in. bbox_inches='tight' changes the saved
        # size, so measure the bytes rather than trusting fig.get_size_inches().
        w = int.from_bytes(png[16:20], "big")
        h = int.from_bytes(png[20:24], "big")
        dpi = float(save_args["dpi"])
        return round(w * 96 / dpi), round(h * 96 / dpi)

    if close_fig:
        plt.close(light_fig)
        plt.close(dark_fig)

    asset_name = name or "figure"

    def _src(data: bytes, theme: str) -> str:
        if publish is not None:
            return publish.asset_url(data, name=f"{asset_name}-{theme}.png")
        return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"

    light_uri = _src(light_png, "light")
    dark_uri = _src(dark_png, "dark")

    escaped_name = html.escape(asset_name)
    # Collapse whitespace so a triple-quoted, indented alt_text reads as one line (it
    # would attribute-encode either way; this just keeps the exported HTML tidy).
    escaped_alt = html.escape(" ".join((alt_text or "Plot").split()))
    # Shrink to fit a narrow viewport (height follows to keep the aspect), but never
    # grow past the physical size set by the width/height attributes.
    style = (
        f"max-width: min(100%, {max_width}); height: auto;"
        if max_width is not None
        else "max-width: 100%; height: auto;"
    )
    escaped_style = html.escape(style)
    # Derived from the asset name (not random) so re-exporting an unchanged report
    # produces byte-identical HTML — a random suffix here would churn the report on
    # every run even though the figures themselves are unchanged.
    class_suffix = hashlib.sha256(asset_name.encode()).hexdigest()[:12]
    figure_class = f"mini-themed-figure-{class_suffix}"
    no_explicit_theme_selector = (
        'body:not([data-theme="dark"]):not([data-theme="light"])'
        ":not(.dark):not(.dark-theme):not(.light):not(.light-theme)"
    )
    css = dedent(f"""
        <style>
        .{figure_class} {{
            .mini-themed-img-dark {{
                display: none;
            }}

            .mini-themed-img-light {{
                display: block;
            }}
        }}

        body[data-theme='dark'],
        body.dark,
        body.dark-theme {{
            .{figure_class} {{
                .mini-themed-img-dark {{
                    display: block;
                }}

                .mini-themed-img-light {{
                    display: none;
                }}
            }}
        }}

        @media (prefers-color-scheme: dark) {{
            {no_explicit_theme_selector} {{
                .{figure_class} {{
                    .mini-themed-img-dark {{
                        display: block;
                    }}

                    .mini-themed-img-light {{
                        display: none;
                    }}
                }}
            }}
        }}

        .{figure_class} > figcaption {{
            margin-top: 0.6em;
            font-size: 0.9em;
            font-style: italic;
            opacity: 0.75;
            text-wrap: balance;
        }}
        </style>
        """)
    light_w, light_h = _css_size(light_png)
    dark_w, dark_h = _css_size(dark_png)
    imgs = dedent(f"""
        <img class="mini-themed-img-light" src="{light_uri}" alt="{escaped_alt}" width="{light_w}" height="{light_h}" style="{escaped_style}" data-asset-name="{escaped_name}" />
        <img class="mini-themed-img-dark" src="{dark_uri}" alt="{escaped_alt}" width="{dark_w}" height="{dark_h}" style="{escaped_style}" data-asset-name="{escaped_name}" />
        """)
    figure = figure_html(imgs, caption=caption, class_=figure_class)

    return f"{css}{figure}"

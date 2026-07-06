"""
Notebook utilities for rendering themed matplotlib figures as HTML.

A report's figures are heavy (a themed plot is *two* PNGs, light and dark). Inlined
as ``data:`` URIs they bloat the exported HTML — the bytes Git LFS used to carry. A
:class:`~mini.reports.Publisher` instead writes each blob out as a file beside the
report (keyed by its readable name) and references it by a **relative** URL, so the
report HTML stays light. Set one up once per report and every ``@themed`` figure
externalizes with no per-figure ceremony::

    # in the report's setup cell
    from mini.vis import themed
    from mini.reports import use_publisher, report_bundle
    use_publisher(report_bundle(__file__))

    # in a figure cell — unchanged
    @themed(alt_text='…')
    def _plot(): ...
    mo.Html(_plot())

The relative reference is the point: the *same* HTML works both ways. Opened locally
it resolves to the co-located ``_assets/`` files (offline, and the figures are real
PNG files); published, ``scripts/build_site.py`` uploads those files to the HF bucket
and inserts a single ``<base href>`` so the very same relative URLs resolve there. A
report with no publisher inlines as self-contained ``data:`` URIs, as before. The
publisher and the bundle protocol live in :mod:`mini.reports`.
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


__all__ = ["themed", "themed_figure_html"]

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
    max_width: str | None = None,
    name: str | None = None,
    publish: Publisher | None = None,
    light_styles: Sequence[Stylesheet] = ("base", "light"),
    dark_styles: Sequence[Stylesheet] = ("base", "dark"),
) -> Callable[P, str] | Callable[[Callable[P, Figure]], Callable[P, str]]:
    """Wrap a plot function to render in both light and dark themes.

    Inside each call, :func:`~mini.vis.plt.use_theme` sets an active
    theme so the plot can use :func:`~mini.vis.plt.light_dark` to
    pick theme-dependent values.

    Can be used as a plain decorator, a decorator factory, or called directly::

        @themed
        def plot(): ...

        @themed(alt_text='My plot')
        def plot(): ...

        themed(plot_lr_finder, alt_text='LR finder')(lr_history, lr_config)

    By default the figure is inlined as a ``data:`` URI. To externalize it (keeping the
    report HTML light), set a default :class:`~mini.reports.Publisher` with
    :func:`~mini.reports.use_publisher`, or pass ``publish=`` one here. *name* is the
    externalized figure's readable basename (it ends up in the asset filename and the
    download name); it defaults to the plot function's name.
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
                max_width=max_width,
                name=name or getattr(fn, "__name__", "").lstrip("_") or "figure",
                publish=publish if publish is not None else current_publisher(),
            )

        return wrapper

    if plot is not None:
        return decorator(plot)
    return decorator


def themed_figure_html(
    light_fig: Figure,
    dark_fig: Figure,
    *,
    close_fig: bool = True,
    alt_text: str | None = None,
    max_width: str | None = None,
    name: str | None = None,
    publish: Publisher | None = None,
    **savefig_kwargs: str | int | bool,
) -> str:
    """Render light/dark matplotlib figures as an HTML figure element.

    With ``publish`` set, each PNG is written out and referenced by a relative URL
    (named ``<name>-light.png`` / ``<name>-dark.png`` so a saved file reads sensibly);
    otherwise both inline as ``data:`` URIs. *name* is also surfaced as a
    ``data-asset-name`` attribute on each ``<img>`` for provenance.
    """
    import base64
    import hashlib
    import html
    from io import BytesIO

    import matplotlib.pyplot as plt

    defaults = {
        "bbox_inches": "tight",
        "dpi": 150,
    }
    save_args = defaults | savefig_kwargs

    def _png_bytes(fig: Figure) -> bytes:
        img_io = BytesIO()
        fig.savefig(img_io, format="png", facecolor=fig.get_facecolor(), **save_args)  # ty:ignore[invalid-argument-type]
        return img_io.getvalue()

    light_png = _png_bytes(light_fig)
    dark_png = _png_bytes(dark_fig)

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
    escaped_alt = html.escape(alt_text or "Plot")
    style = f"max-width: {max_width};" if max_width is not None else ""
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
        </style>
        """)
    figure_html = dedent(f"""
        <figure class="{figure_class}">
            <img class="mini-themed-img-light" src="{light_uri}" alt="{escaped_alt}" style="{escaped_style}" data-asset-name="{escaped_name}" />
            <img class="mini-themed-img-dark" src="{dark_uri}" alt="{escaped_alt}" style="{escaped_style}" data-asset-name="{escaped_name}" />
        </figure>
        """)

    return f"{css}{figure_html}"

"""
Notebook utilities for rendering themed matplotlib figures as HTML.

A report's figures are heavy (a themed plot is *two* PNGs, light and dark). Inlined
as ``data:`` URIs they bloat the exported HTML — the bytes Git LFS used to carry.
A :class:`Publisher` instead writes each blob out as a content-addressed file beside
the report and references it by a **relative** URL, so the report HTML stays light.
Set one up once per report and every ``@themed`` figure externalizes with no
per-figure ceremony::

    # in the report's setup cell
    from mini.vis import themed, use_publisher, report_bundle
    use_publisher(report_bundle(__file__))

    # in a figure cell — unchanged
    @themed(alt_text='…')
    def _plot(): ...
    mo.Html(_plot())

The relative reference is the point: the *same* HTML works both ways. Opened locally
it resolves to the co-located ``_assets/`` files (offline, and the figures are real
PNG files); published, ``scripts/build_site.py`` uploads those files to the HF bucket
and inserts a single ``<base href>`` so the very same relative URLs resolve there. A
report with no publisher inlines as self-contained ``data:`` URIs, as before.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from functools import wraps
from pathlib import Path, PurePosixPath
from textwrap import dedent
from typing import Callable, ParamSpec, TypeVar, overload

from .plt import use_style
from .theme import use_theme

from collections.abc import Sequence

from matplotlib.figure import Figure

from mini.vis.plt import Stylesheet


__all__ = ['themed', 'themed_figure_html', 'Publisher', 'use_publisher', 'report_bundle']

P = ParamSpec('P')
R = TypeVar('R')


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Publishing report assets as files referenced by a relative URL
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Publisher:
    """Writes a report's heavy assets out as content-addressed files beside the
    exported HTML, referenced by a **relative** URL.

    Each blob is written once, keyed by its SHA-256, under ``asset_dir``
    (conventionally ``…/__marimo__/_assets``); the reference is ``<link>/<sha><ext>``.
    Because it's relative, the same HTML resolves to the local files when opened off
    disk and to the HF bucket when published (a single ``<base href>`` is inserted at
    build time — see ``scripts/build_site.py``). Use :meth:`asset_url` for any blob (a
    JSON a data-browser widget fetches, an SPA's data files); :meth:`png_url` is the
    figure-shaped wrapper :func:`themed` calls.
    """

    asset_dir: Path
    link: str = '_assets'

    def asset_url(self, data: bytes | Path, *, name: str) -> str:
        """Write *data* (bytes or a file) as ``<sha><ext>`` and return its relative URL.

        Content-addressed and write-once, so re-running a report reuses identical
        bytes rather than piling up copies. *name* only supplies the extension (which
        sets the served media type); the SHA names the file.
        """
        blob = bytes(data) if isinstance(data, (bytes, bytearray)) else Path(data).read_bytes()
        filename = f'{hashlib.sha256(blob).hexdigest()}{PurePosixPath(name).suffix}'
        dest = self.asset_dir / filename
        if not dest.exists():  # content-addressed: identical bytes already written
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f'{filename}.tmp')
            tmp.write_bytes(blob)
            tmp.replace(dest)  # atomic, so a concurrent reader never sees a partial file
        return f'{self.link}/{filename}'

    def png_url(self, data: bytes) -> str:
        """Write a PNG and return its relative URL (the figure-shaped :meth:`asset_url`)."""
        return self.asset_url(data, name='figure.png')


def report_bundle(notebook_file: str | Path, *, link: str = '_assets') -> Publisher:
    """A :class:`Publisher` writing assets beside a report's exported HTML.

    Marimo exports ``docs/<…>/report.py`` to ``docs/<…>/__marimo__/report.html``; this
    points assets at ``docs/<…>/__marimo__/<link>/`` so the relative ``<link>/…`` URL
    resolves next to that HTML. Call it from the report's setup cell with ``__file__``::

        use_publisher(report_bundle(__file__))
    """
    out_dir = Path(notebook_file).resolve().parent / '__marimo__'
    return Publisher(asset_dir=out_dir / link, link=link)


_default_publisher: Publisher | None = None


def use_publisher(publisher: Publisher | None) -> Publisher | None:
    """Set the report-wide default publisher; call once in a report's setup cell.

    Every ``@themed`` figure then externalizes through it with no per-figure argument.
    Pass a :class:`Publisher` (usually from :func:`report_bundle`), or ``None`` to clear
    it (figures inline as self-contained ``data:`` URIs). Returns it, e.g. to call
    :meth:`~Publisher.asset_url` for a data blob.
    """
    global _default_publisher
    _default_publisher = publisher
    return publisher


# ---------------------------------------------------------------------------
# Themed figures
# ---------------------------------------------------------------------------


@overload
def themed(
    plot: Callable[P, Figure],
    *,
    alt_text: str | None = ...,
    max_width: str | None = ...,
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
    publish: Publisher | None = ...,
    light_styles: Sequence[Stylesheet] = ...,
    dark_styles: Sequence[Stylesheet] = ...,
) -> Callable[[Callable[P, Figure]], Callable[P, str]]: ...


def themed(
    plot: Callable[P, Figure] | None = None,
    *,
    alt_text: str | None = None,
    max_width: str | None = None,
    publish: Publisher | None = None,
    light_styles: Sequence[Stylesheet] = ('base', 'light'),
    dark_styles: Sequence[Stylesheet] = ('base', 'dark'),
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
    report HTML light), set a default :class:`Publisher` with :func:`use_publisher`, or
    pass ``publish=`` one here.
    """

    def decorator(fn: Callable[P, Figure]) -> Callable[P, str]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
            with use_theme('light'), use_style(*light_styles):
                light_fig = fn(*args, **kwargs)
            with use_theme('dark'), use_style(*dark_styles):
                dark_fig = fn(*args, **kwargs)

            if light_fig is None or dark_fig is None:
                msg = f'{fn.__name__} returned None'
                raise ValueError(msg)

            return themed_figure_html(
                light_fig,
                dark_fig,
                alt_text=alt_text,
                max_width=max_width,
                publish=publish if publish is not None else _default_publisher,
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
    publish: Publisher | None = None,
    **savefig_kwargs: str | int | bool,
) -> str:
    """Render light/dark matplotlib figures as an HTML figure element.

    With ``publish`` set, each PNG is written out and referenced by a relative URL;
    otherwise both inline as ``data:`` URIs.
    """
    import base64
    import html
    import secrets
    from io import BytesIO

    import matplotlib.pyplot as plt

    defaults = {
        'bbox_inches': 'tight',
        'dpi': 150,
    }
    save_args = defaults | savefig_kwargs

    def _png_bytes(fig: Figure) -> bytes:
        img_io = BytesIO()
        fig.savefig(img_io, format='png', facecolor=fig.get_facecolor(), **save_args)  # ty:ignore[invalid-argument-type]
        return img_io.getvalue()

    light_png = _png_bytes(light_fig)
    dark_png = _png_bytes(dark_fig)

    if close_fig:
        plt.close(light_fig)
        plt.close(dark_fig)

    def _src(data: bytes) -> str:
        if publish is not None:
            return publish.png_url(data)
        return f'data:image/png;base64,{base64.b64encode(data).decode("ascii")}'

    light_uri = _src(light_png)
    dark_uri = _src(dark_png)

    escaped_alt = html.escape(alt_text or 'Plot')
    style = f'max-width: {max_width};' if max_width is not None else ''
    escaped_style = html.escape(style)
    class_suffix = secrets.token_hex(6)
    figure_class = f'mini-themed-figure-{class_suffix}'
    no_explicit_theme_selector = (
        'body:not([data-theme="dark"]):not([data-theme="light"])'
        ':not(.dark):not(.dark-theme):not(.light):not(.light-theme)'
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
            <img class="mini-themed-img-light" src="{light_uri}" alt="{escaped_alt}" style="{escaped_style}" />
            <img class="mini-themed-img-dark" src="{dark_uri}" alt="{escaped_alt}" style="{escaped_style}" />
        </figure>
        """)

    return f'{css}{figure_html}'

"""
Notebook utilities for rendering themed matplotlib figures as HTML.

A report's figures are heavy (a themed plot is *two* PNGs, light and dark). Inlined
as ``data:`` URIs they bloat the exported HTML — the bytes Git LFS used to carry.
A :class:`Publisher` routes them through the artifact :class:`~mini.store.Store`
instead: each PNG is ``put`` + ``publish``ed and the ``<img>`` points at its web
URL, so the report HTML stays light enough to live in Git. Set one up once per
report and every ``@themed`` figure externalizes with no per-figure ceremony::

    # in the report's setup cell
    from mini import LocalApparatus
    from mini.vis import themed, use_publisher
    use_publisher(LocalApparatus(NAME).store(), prefix=f'reports/{NAME}')

    # in a figure cell — unchanged
    @themed(alt_text='…')
    def _plot(): ...
    mo.Html(_plot())

Without a publisher (or with a local-only store that can't serve web URLs) figures
fall back to inlining, so an offline ``./go build`` still renders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path, PurePosixPath
from textwrap import dedent
from typing import Callable, ParamSpec, TypeVar, overload

from .plt import use_style
from .theme import use_theme

from collections.abc import Sequence

from matplotlib.figure import Figure

from mini.store import Store
from mini.vis.plt import Stylesheet


__all__ = ['themed', 'themed_figure_html', 'Publisher', 'use_publisher']

P = ParamSpec('P')
R = TypeVar('R')


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Publishing report assets to web URLs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Publisher:
    """Routes report assets (figures, data blobs) to a :class:`~mini.store.Store`'s
    published web URLs, under a shared ``prefix``.

    The same content-addressed ``publish`` the store uses for any artifact, scoped
    to a report's asset namespace. Use :meth:`png_url` for themed figures (it signals
    a graceful inline fallback) and :meth:`asset_url` for arbitrary blobs a report's
    JS reads — a large JSON a data-browser widget fetches, an SPA's data files.
    """

    store: Store | None = None
    prefix: str = 'reports'

    def asset_url(self, data: bytes | Path, *, name: str) -> str | None:
        """Publish bytes (or a file) at ``<prefix>/<name>`` and return its URL.

        ``None`` only when no store is configured. A web-serving backend (an HF
        bucket) returns an ``https://`` URL the browser can fetch cross-origin; a
        local store returns a ``file://`` URL — fine when opening the HTML straight
        off disk, but it won't fetch from a *served* page, which is why figures
        inline instead of relying on it (see :meth:`png_url`).
        """
        if self.store is None:
            return None
        art = self.store.put(data, name=PurePosixPath(name).name)
        return self.store.publish(art, f'{self.prefix}/{name}')

    def png_url(self, data: bytes, *, slug: str) -> str | None:
        """Publish a PNG and return a *web* URL, or ``None`` to inline it instead.

        Narrower than :meth:`asset_url` on purpose: an ``<img>`` only benefits from
        externalizing if the URL actually serves over the wire, so a missing store
        *or* a non-web (``file://``) store both yield ``None`` — the caller inlines.
        """
        url = self.asset_url(data, name=f'{slug}.png')
        return url if (url and url.startswith('https://')) else None


def _as_publisher(publisher: Publisher | Store | None) -> Publisher | None:
    if isinstance(publisher, Publisher):
        return publisher
    return Publisher(publisher) if publisher is not None else None


_default_publisher: Publisher | None = None


def use_publisher(publisher: Publisher | Store | None, *, prefix: str | None = None) -> Publisher | None:
    """Set the report-wide default publisher; call once in a report's setup cell.

    Every ``@themed`` figure then externalizes through it with no per-figure
    argument. Pass a :class:`~mini.store.Store` (optionally with ``prefix``) for the
    common case, a :class:`Publisher` to reuse a configured one, or ``None`` to clear
    it (figures inline). Returns the resolved publisher (e.g. to call
    :meth:`~Publisher.asset_url` on it for a data blob).
    """
    global _default_publisher
    pub = _as_publisher(publisher)
    if pub is not None and prefix is not None:
        pub = replace(pub, prefix=prefix)
    _default_publisher = pub
    return pub


# ---------------------------------------------------------------------------
# Themed figures
# ---------------------------------------------------------------------------


@overload
def themed(
    plot: Callable[P, Figure],
    *,
    alt_text: str | None = ...,
    max_width: str | None = ...,
    publish: Publisher | Store | None = ...,
    name: str | None = ...,
    light_styles: Sequence[Stylesheet] = ...,
    dark_styles: Sequence[Stylesheet] = ...,
) -> Callable[P, str]: ...


@overload
def themed(
    plot: None = ...,
    *,
    alt_text: str | None = ...,
    max_width: str | None = ...,
    publish: Publisher | Store | None = ...,
    name: str | None = ...,
    light_styles: Sequence[Stylesheet] = ...,
    dark_styles: Sequence[Stylesheet] = ...,
) -> Callable[[Callable[P, Figure]], Callable[P, str]]: ...


def themed(
    plot: Callable[P, Figure] | None = None,
    *,
    alt_text: str | None = None,
    max_width: str | None = None,
    publish: Publisher | Store | None = None,
    name: str | None = None,
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

    By default the figure is inlined as a ``data:`` URI. To externalize it to a web
    URL (and keep the report HTML light), set a default with :func:`use_publisher`,
    or pass ``publish=`` a :class:`Publisher`/:class:`~mini.store.Store` here. ``name``
    gives the figure a stable URL slug; omitted, a content hash names it.
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

            pub = _as_publisher(publish) if publish is not None else _default_publisher
            return themed_figure_html(
                light_fig,
                dark_fig,
                alt_text=alt_text,
                max_width=max_width,
                publish=pub,
                name=name,
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
    name: str | None = None,
    **savefig_kwargs: str | int | bool,
) -> str:
    """Render light/dark matplotlib figures as an HTML figure element.

    With ``publish`` set, each PNG is externalized to a web URL (falling back to an
    inline ``data:`` URI when the store can't serve one); otherwise both inline.
    """
    import base64
    import hashlib
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

    # One slug base groups the light/dark pair; a content hash keeps the URL stable
    # by content when the author doesn't name it.
    base = name or hashlib.sha256(light_png + dark_png).hexdigest()[:12]

    def _src(data: bytes, slug: str) -> str:
        if publish is not None and (url := publish.png_url(data, slug=slug)):
            return url
        return f'data:image/png;base64,{base64.b64encode(data).decode("ascii")}'

    light_uri = _src(light_png, f'{base}-light')
    dark_uri = _src(dark_png, f'{base}-dark')

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

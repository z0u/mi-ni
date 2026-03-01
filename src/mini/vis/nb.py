"""
Notebook utilities for rendering themed matplotlib figures as HTML.
"""

from __future__ import annotations

import logging
from functools import wraps
from textwrap import dedent
from typing import Callable, ParamSpec, TypeVar

from .plt import use_style
from .theme import use_theme

from collections.abc import Sequence

from matplotlib.figure import Figure

from mini.vis.plt import Stylesheet


__all__ = ['themed', 'themed_figure_html']

P = ParamSpec('P')
R = TypeVar('R')


log = logging.getLogger(__name__)


def themed(
    plot: Callable[P, Figure],
    *,
    alt_text: str | None = None,
    max_width: str | None = None,
    light_styles: Sequence[Stylesheet] = ('base', 'light'),
    dark_styles: Sequence[Stylesheet] = ('base', 'dark'),
) -> Callable[P, str]:
    """Wrap a plot function to render in both light and dark themes.

    Inside each call, :func:`~mini.vis.plt.use_theme` sets an active
    theme so the plot can use :func:`~mini.vis.plt.light_dark` to
    pick theme-dependent values.

    Returns a wrapper; call it with the original arguments::

        themed(plot_lr_finder)(lr_history, lr_config)
    """

    @wraps(plot)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        with use_theme('light'), use_style(*light_styles):
            light_fig = plot(*args, **kwargs)
        with use_theme('dark'), use_style(*dark_styles):
            dark_fig = plot(*args, **kwargs)

        if light_fig is None or dark_fig is None:
            msg = f'{plot.__name__} returned None'
            raise ValueError(msg)

        return themed_figure_html(
            light_fig,
            dark_fig,
            alt_text=alt_text,
            max_width=max_width,
        )

    return wrapper


def themed_figure_html(
    light_fig: Figure,
    dark_fig: Figure,
    *,
    close_fig: bool = True,
    alt_text: str | None = None,
    max_width: str | None = None,
    **savefig_kwargs: str | int | bool,
) -> str:
    """Render light/dark matplotlib figures as an HTML figure element."""
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

    def _to_data_uri(fig: Figure) -> str:
        img_io = BytesIO()
        fig.savefig(img_io, format='png', facecolor=fig.get_facecolor(), **save_args)  # ty:ignore[invalid-argument-type]
        payload = base64.b64encode(img_io.getvalue()).decode('ascii')
        return f'data:image/png;base64,{payload}'

    light_uri = _to_data_uri(light_fig)
    dark_uri = _to_data_uri(dark_fig)

    if close_fig:
        plt.close(light_fig)
        plt.close(dark_fig)

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

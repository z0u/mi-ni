import logging
from textwrap import dedent
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from matplotlib.figure import Figure


__all__ = ['save_fig', 'themed_figure_html']


log = logging.getLogger(__name__)


def save_fig(
    fig: 'Figure',
    filepath: str | Path,
    close_fig: bool = True,
    alt_text: str | None = None,
    max_width: str | None = '70rem',
    **savefig_kwargs,
) -> str:
    """
    Save a matplotlib Figure to a file and returns an HTML img tag string.

    Ensures the target directory exists, saves the figure with sensible defaults
    (like tight bounding box and matching facecolor), optionally closes the
    figure object, and adds a cache-busting query parameter to the img src.

    Args:
        fig: The matplotlib Figure object.
        filepath: The path (including filename and extension) to save the figure.
        close_fig: Whether to close the figure object after saving (prevents potential double display in some environments).
        alt_text: Optional alt text for the HTML img tag. Defaults to a generic message.
        max_width: Optional CSS max-width for the img tag.
        **savefig_kwargs: Additional keyword arguments passed to fig.savefig().
                          Defaults include facecolor=fig.get_facecolor(),
                          bbox_inches='tight', and dpi=150.

    Returns:
        An HTML string '<img src="..." alt="...">'.
    """
    import html
    import secrets
    import urllib.parse

    import matplotlib.pyplot as plt

    filepath = Path(filepath)
    log.debug(f"Saving figure to '{filepath}'")

    # Ensure the parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Sensible defaults for savefig, allowing user overrides
    defaults = {
        'facecolor': fig.get_facecolor(),
        'bbox_inches': 'tight',
        'dpi': 150,  # A good balance for notebooks
    }
    save_args = defaults | savefig_kwargs

    # Save the figure
    fig.savefig(filepath, **save_args)
    log.debug(f"Figure saved: '{filepath}'")

    if close_fig:
        plt.close(fig)
        log.debug(f"Closed figure object for '{filepath}'")

    # Use figure filename as default alt text if not provided
    if alt_text is None:
        alt_text = f'Plot saved at {filepath.name}'

    style = f'max-width: {max_width};' if max_width is not None else ''

    escaped_alt = html.escape(alt_text)
    cache_buster = secrets.token_urlsafe()
    safe_src = urllib.parse.quote(filepath.as_posix())
    escaped_style = html.escape(style)
    return f'<img src="{safe_src}?v={cache_buster}" alt="{escaped_alt}" style="{escaped_style}" />'


def themed_figure_html(
    light_fig: 'Figure',
    dark_fig: 'Figure',
    *,
    close_fig: bool = True,
    alt_text: str | None = None,
    max_width: str | None = '70rem',
    **savefig_kwargs,
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

    def _to_data_uri(fig: 'Figure') -> str:
        img_io = BytesIO()
        fig.savefig(img_io, format='png', facecolor=fig.get_facecolor(), **save_args)
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

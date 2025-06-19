import logging
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    # This import is only needed for type hinting
    from matplotlib.figure import Figure


log = logging.getLogger(__name__)


def displayer():
    import secrets

    handle = f'displayer-{secrets.token_hex(16)}'
    first = True

    def show(ob):
        nonlocal handle, first
        from IPython.display import display, update_display

        if first:
            first = False
            display(ob, display_id=handle)
        else:
            update_display(ob, display_id=handle)

    return show


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

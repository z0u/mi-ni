"""
Utilities for working with matplotlib stylesheets.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt


Stylesheet = Literal["base", "light", "dark", "transparent"] | Mapping[str, str]


@contextmanager
def use_style(*styles: Stylesheet):
    """Apply matplotlib styles.

    When *theme* is given, :func:`light_dark` and :func:`current_theme`
    will resolve against it inside the block.
    """
    with mpl.rc_context():
        stylesheet_dir = Path(__file__).parent / "mplstyles"
        for style in styles:
            if isinstance(style, Mapping):
                plt.style.use(dict(style))
            else:
                plt.style.use(stylesheet_dir / f"{style}.mplstyle")
        yield

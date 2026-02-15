from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, Literal, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


Stylesheet = Literal['base', 'light', 'dark', 'transparent'] | Mapping[str, str]
ThemeName = Literal['light', 'dark', 'indeterminate']


@contextmanager
def use_theme(*styles: Stylesheet):
    with mpl.rc_context():
        stylesheet_dir = Path(__file__).parent / 'mplstyles'
        for style in styles:
            if isinstance(style, Mapping):
                plt.style.use(dict(style))
            else:
                plt.style.use(stylesheet_dir / f'{style}.mplstyle')
        yield


@dataclass
class Theme:
    name: ThemeName
    """The name of the this theme."""

    def __init__(self, name: ThemeName):
        self.name = name

    def val[T](
        self,
        default: T,
        *,
        light: T | None = None,
        dark: T | None = None,
        indeterminate: T | None = None,
    ) -> T:
        """
        Select a value based on the theme.

        Args:
            light: The value to use if this is a light theme.

            dark: The value to use if this is a dark theme.

            indeterminate: The value to use if this theme is neither light nor
            dark.

            default: The value to use if a value has not been provided for this
            theme.

        Returns:
            The value passed for the keyword argument matching the theme name,
            or `default` if that value was `None`.
        """
        if self.name == 'light' and light is not None:
            return light
        if self.name == 'dark' and dark is not None:
            return dark
        if self.name == 'indeterminate' and indeterminate is not None:
            return indeterminate
        return default


def autoclose(factory: Callable[..., Figure | None]) -> Callable[..., Figure | None]:
    @wraps(factory)
    def _autoclose(*args, **kwargs) -> Figure | None:
        fig = factory(*args, **kwargs)
        plt.close(fig)
        return fig

    return _autoclose

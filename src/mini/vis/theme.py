"""
Utilities for working with themes in mini. This is used by the notebook utilities to pick values based on the active theme.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

ThemeName = Literal["light", "dark"]

_current_theme: ContextVar[ThemeName] = ContextVar("_current_theme", default="light")


@contextmanager
def use_theme(theme: ThemeName):
    """Set the active theme for the block. This is used by :func:`light_dark` to pick values based on the theme.

    When *theme* is given, :func:`light_dark` and :func:`current_theme`
    will resolve against it inside the block.
    """
    token = _current_theme.set(theme)
    try:
        yield
    finally:
        _current_theme.reset(token)


def current_theme() -> ThemeName:
    """Return the active theme name."""
    return _current_theme.get()


def light_dark[T](light: T, dark: T) -> T:
    """Pick a value based on the active theme (like CSS ``light-dark()``)."""
    return dark if _current_theme.get() == "dark" else light

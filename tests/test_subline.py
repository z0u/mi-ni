"""Subline rendering: line-wrap arithmetic and the custom-CSS hook."""

import numpy as np
import pytest

from subline.series import Series
from subline.subline import Subline


def _series(n: int) -> Series:
    return Series(raw=np.linspace(0, 1, n), label="s")


@pytest.mark.parametrize("n", [1, 5, 10, 20, 40, 47, 80])
def test_exact_fit_stays_on_one_line(n):
    """`n` characters at `chars_per_line=n` fit on a single line.

    Regression for a float-accumulation bug: summing char widths one token at a time
    drifted past `n * char_width` (e.g. 10 * 8.4), wrapping a line one character early,
    so callers padded `chars_per_line` by one to compensate.
    """
    sub = Subline(chars_per_line=n)
    lines = sub._wrap_tokens(sub._get_token_spans(list("x" * n)))
    assert len(lines) == 1


@pytest.mark.parametrize("n", [1, 10, 20, 80])
def test_one_char_over_wraps(n):
    """One character past the budget wraps to a second line — the tolerance is sub-character."""
    sub = Subline(chars_per_line=n)
    lines = sub._wrap_tokens(sub._get_token_spans(list("x" * (n + 1))))
    assert len(lines) == 2


def test_custom_css_overrides_defaults():
    """`css` is appended after the built-in styles, so a later rule wins at equal specificity."""
    svg = Subline(chars_per_line=20, css="svg { --bg-color: red; }").plot("hello", [_series(5)])
    assert "--col-series-1" in svg  # base theme still present
    assert svg.index("--bg-color: red") > svg.index("--bg-color: light-dark")  # override comes last


def test_default_css_is_neutral():
    """Absent a `css` override, the library keeps its own neutral dark background."""
    svg = Subline().plot("hi", [_series(2)])
    assert "light-dark(#fff, #2a2a2a)" in svg

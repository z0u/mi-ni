"""Pre-computed colors stand in for alpha, so pin that they land where alpha would have."""

import pytest

from mini.vis import mix, page_color, use_theme


@pytest.mark.parametrize("theme,expected", [("light", "#ffffff"), ("dark", "#111111")])
def test_the_page_follows_the_theme(theme, expected):
    with use_theme(theme):
        assert page_color() == expected


def test_mix_runs_from_one_color_to_the_other():
    assert mix("#000000", "#ffffff", 0.0) == "#000000"
    assert mix("#000000", "#ffffff", 1.0) == "#ffffff"
    assert mix("#000000", "#ffffff", 0.5) == "#808080"


def test_mix_lands_where_alpha_would_have():
    """Matplotlib composites in sRGB, so this is the same arithmetic it does at draw time."""
    assert mix("#ffffff", "#444444", 0.5) == mix("#444444", "#ffffff", 0.5)
    assert mix("#ffffff", "#000000", 0.25) == "#bfbfbf"  # 0.75 of the way to white

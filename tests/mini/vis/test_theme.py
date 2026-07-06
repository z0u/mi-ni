import pytest
from mini.vis.theme import ThemeName, current_theme, light_dark, use_theme


@pytest.mark.parametrize(
    "theme,value",
    [
        ("light", "a"),
        ("dark", "b"),
    ],
)
def test_light_dark(theme: ThemeName, value: str):
    with use_theme(theme):
        assert current_theme() == theme
        assert light_dark("a", "b") == value


def test_default_theme():
    assert current_theme() == "light"
    assert light_dark("a", "b") == "a"

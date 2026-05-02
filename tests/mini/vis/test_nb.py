from unittest.mock import patch

from mini.vis.theme import light_dark
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mini.vis.nb import themed

matplotlib.use('Agg')


def _dummy_plot(x: int, y: int) -> Figure:
    """Minimal plot function that uses light_dark."""
    fig, ax = plt.subplots()
    ax.plot([0, x], [0, y])
    ax.set_facecolor(light_dark('#fff', '#000'))
    return fig


def test_html_contains_both_variants():
    result = themed(_dummy_plot)(1, 2)
    assert 'mini-themed-img-light' in result
    assert 'mini-themed-img-dark' in result


def test_themed_value_is_used():
    """Verify set_facecolor receives both light and dark values."""
    original = Axes.set_facecolor
    seen: list[str] = []

    def spy(self, color):
        seen.append(color)
        return original(self, color)

    with patch.object(Axes, 'set_facecolor', spy):
        themed(_dummy_plot)(1, 2)

    assert '#fff' in seen
    assert '#000' in seen


def test_alt_text():
    result = themed(_dummy_plot, alt_text='My plot')(1, 2)
    assert 'alt="My plot"' in result


def test_decorator_factory():
    @themed(alt_text='Factory plot')
    def plot(x: int) -> Figure:
        fig, _ = plt.subplots()
        return fig

    result = plot(1)
    assert 'alt="Factory plot"' in result
    assert 'mini-themed-img-light' in result

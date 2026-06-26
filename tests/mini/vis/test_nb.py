from pathlib import Path
from unittest.mock import patch

from mini.vis.theme import light_dark
import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mini.store import LocalStore
from mini.vis.nb import Publisher, themed, use_publisher

matplotlib.use('Agg')


class _WebStore(LocalStore):
    """A LocalStore that pretends to serve over the web (https publish URLs)."""

    def publish(self, art, path):  # type: ignore[override]
        return f'https://buckets.example/resolve/published/{path}'


@pytest.fixture(autouse=True)
def _clear_default_publisher():
    """Keep the module-level default from leaking between tests."""
    use_publisher(None)
    yield
    use_publisher(None)


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


def test_default_inlines_as_data_uri():
    result = themed(_dummy_plot)(1, 2)
    assert result.count('src="data:image/png;base64,') == 2
    assert 'https://' not in result


def test_publish_externalizes_to_web_urls(tmp_path: Path):
    pub = Publisher(_WebStore(tmp_path / 'store'), prefix='reports/demo')
    result = themed(_dummy_plot, publish=pub)(1, 2)
    # Both variants reference web URLs, not inline data, and stay grouped by slug.
    assert 'src="data:image' not in result
    assert 'https://buckets.example/resolve/published/reports/demo/' in result
    assert '-light.png' in result
    assert '-dark.png' in result


def test_named_figure_gives_stable_slug(tmp_path: Path):
    pub = Publisher(_WebStore(tmp_path / 'store'))
    result = themed(_dummy_plot, publish=pub, name='loss-curve')(1, 2)
    assert 'reports/loss-curve-light.png' in result
    assert 'reports/loss-curve-dark.png' in result


def test_local_store_falls_back_to_inline(tmp_path: Path):
    # A non-web store can't serve an <img> over the wire, so figures inline.
    pub = Publisher(LocalStore(tmp_path / 'store'))
    result = themed(_dummy_plot, publish=pub)(1, 2)
    assert result.count('src="data:image/png;base64,') == 2


def test_use_publisher_default_is_picked_up(tmp_path: Path):
    use_publisher(_WebStore(tmp_path / 'store'), prefix='reports/auto')
    result = themed(_dummy_plot)(1, 2)  # no per-figure publish=
    assert 'https://buckets.example/resolve/published/reports/auto/' in result


def test_asset_url_publishes_arbitrary_bytes(tmp_path: Path):
    pub = Publisher(_WebStore(tmp_path / 'store'), prefix='reports/demo')
    url = pub.asset_url(b'{"hello": "world"}', name='data/points.json')
    assert url == 'https://buckets.example/resolve/published/reports/demo/data/points.json'


def test_asset_url_without_store_is_none():
    assert Publisher(None).asset_url(b'x', name='a.json') is None

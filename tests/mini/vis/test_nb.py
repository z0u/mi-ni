import re
from pathlib import Path
from unittest.mock import patch

from mini.vis.theme import light_dark
import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mini.vis.nb import Publisher, report_bundle, themed, use_publisher

matplotlib.use('Agg')


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
    assert '_assets/' not in result


def test_publish_externalizes_to_relative_urls(tmp_path: Path):
    pub = Publisher(tmp_path / '__marimo__' / '_assets')
    result = themed(_dummy_plot, publish=pub)(1, 2)
    # Both variants reference relative _assets/ URLs, not inline data.
    assert 'src="data:image' not in result
    srcs = re.findall(r'src="([^"]+)"', result)
    assert len(srcs) == 2
    assert all(re.fullmatch(r'_assets/[0-9a-f]{64}\.png', s) for s in srcs), srcs
    # …and the referenced files actually exist on disk and are valid PNGs.
    for s in srcs:
        f = tmp_path / '__marimo__' / s
        assert f.exists() and f.read_bytes()[:4] == b'\x89PNG'


def test_distinct_light_dark_files(tmp_path: Path):
    pub = Publisher(tmp_path / '_assets')
    result = themed(_dummy_plot)(1, 2)  # inline (no publisher) → control
    assert 'data:image' in result
    out = themed(_dummy_plot, publish=pub)(1, 2)
    srcs = set(re.findall(r'src="([^"]+)"', out))
    assert len(srcs) == 2  # light and dark hash differently → two files


def test_content_addressed_write_once(tmp_path: Path):
    pub = Publisher(tmp_path)
    u1 = pub.asset_url(b'same-bytes', name='a.json')
    u2 = pub.asset_url(b'same-bytes', name='b.json')
    assert u1 == u2  # identical content → identical URL, written once
    assert len(list(tmp_path.glob('*.json'))) == 1


def test_use_publisher_default_is_picked_up(tmp_path: Path):
    use_publisher(Publisher(tmp_path / '_assets'))
    result = themed(_dummy_plot)(1, 2)  # no per-figure publish=
    assert re.search(r'src="_assets/[0-9a-f]{64}\.png"', result)


def test_asset_url_writes_file_and_returns_relative_url(tmp_path: Path):
    pub = Publisher(tmp_path / '_assets')
    url = pub.asset_url(b'{"hello": "world"}', name='points.json')
    assert re.fullmatch(r'_assets/[0-9a-f]{64}\.json', url)
    assert (tmp_path / url).read_bytes() == b'{"hello": "world"}'


def test_report_bundle_targets_marimo_dir():
    pub = report_bundle('/proj/docs/probe/report.py')
    assert pub.asset_dir == Path('/proj/docs/probe/__marimo__/_assets')
    assert pub.link == '_assets'

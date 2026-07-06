import re
from pathlib import Path
from unittest.mock import patch

from mini.vis.theme import light_dark
import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mini.reports import Publisher, report_bundle, use_publisher
from mini.vis.nb import themed

matplotlib.use("Agg")


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
    ax.set_facecolor(light_dark("#fff", "#000"))
    return fig


def test_html_contains_both_variants():
    result = themed(_dummy_plot)(1, 2)
    assert "mini-themed-img-light" in result
    assert "mini-themed-img-dark" in result


def test_themed_value_is_used():
    """Verify set_facecolor receives both light and dark values."""
    original = Axes.set_facecolor
    seen: list[str] = []

    def spy(self, color):
        seen.append(color)
        return original(self, color)

    with patch.object(Axes, "set_facecolor", spy):
        themed(_dummy_plot)(1, 2)

    assert "#fff" in seen
    assert "#000" in seen


def test_alt_text():
    result = themed(_dummy_plot, alt_text="My plot")(1, 2)
    assert 'alt="My plot"' in result


def test_decorator_factory():
    @themed(alt_text="Factory plot")
    def plot(x: int) -> Figure:
        fig, _ = plt.subplots()
        return fig

    result = plot(1)
    assert 'alt="Factory plot"' in result
    assert "mini-themed-img-light" in result


def test_default_inlines_as_data_uri():
    result = themed(_dummy_plot)(1, 2)
    assert result.count('src="data:image/png;base64,') == 2
    assert "_assets/" not in result


def test_publish_externalizes_to_relative_urls(tmp_path: Path):
    pub = Publisher(tmp_path / "__marimo__" / "_assets")
    result = themed(_dummy_plot, publish=pub)(1, 2)
    # Both variants reference relative _assets/ URLs, not inline data.
    assert 'src="data:image' not in result
    srcs = re.findall(r'src="([^"]+)"', result)
    assert len(srcs) == 2
    # Path is the readable leaf (derived from the plot fn name) — the stable URL and
    # what a "Save as" suggests; no content hash in the path.
    assert all(re.fullmatch(r"_assets/dummy_plot-(light|dark)\.png", s) for s in srcs), srcs
    # …and the referenced files actually exist on disk and are valid PNGs.
    for s in srcs:
        f = tmp_path / "__marimo__" / s
        assert f.exists() and f.read_bytes()[:4] == b"\x89PNG"


def test_name_overrides_the_asset_leaf(tmp_path: Path):
    pub = Publisher(tmp_path / "_assets")
    result = themed(_dummy_plot, name="loss-curve", publish=pub)(1, 2)
    srcs = re.findall(r'src="([^"]+)"', result)
    assert all(re.fullmatch(r"_assets/loss-curve-(light|dark)\.png", s) for s in srcs), srcs
    # The readable name is also surfaced for provenance.
    assert result.count('data-asset-name="loss-curve"') == 2


def test_distinct_light_dark_files(tmp_path: Path):
    pub = Publisher(tmp_path / "_assets")
    result = themed(_dummy_plot)(1, 2)  # inline (no publisher) → control
    assert "data:image" in result
    out = themed(_dummy_plot, publish=pub)(1, 2)
    srcs = set(re.findall(r'src="([^"]+)"', out))
    assert len(srcs) == 2  # distinct -light/-dark names → two files


def test_stable_url_overwrites_in_place(tmp_path: Path):
    # A re-export is a fresh Publisher over the same dir; the same name overwrites.
    u1 = Publisher(tmp_path).asset_url(b"first", name="pts.json")
    u2 = Publisher(tmp_path).asset_url(b"second", name="pts.json")
    assert u1 == u2 == "_assets/pts.json"  # stable URL, no content hash
    assert len(list(tmp_path.rglob("*.json"))) == 1  # one file, overwritten
    assert (tmp_path / "pts.json").read_bytes() == b"second"


def test_distinct_blobs_same_name_in_one_report_raise(tmp_path: Path):
    import pytest

    pub = Publisher(tmp_path)
    pub.asset_url(b"first", name="fig.png")
    pub.asset_url(b"first", name="fig.png")  # same bytes again is fine (idempotent)
    with pytest.raises(ValueError, match="distinct"):
        pub.asset_url(b"different", name="fig.png")  # two different figures, one name


def test_use_publisher_default_is_picked_up(tmp_path: Path):
    use_publisher(Publisher(tmp_path / "_assets"))
    result = themed(_dummy_plot)(1, 2)  # no per-figure publish=
    assert re.search(r'src="_assets/dummy_plot-light\.png"', result)


def test_asset_url_writes_file_and_returns_relative_url(tmp_path: Path):
    pub = Publisher(tmp_path / "_assets")
    url = pub.asset_url(b'{"hello": "world"}', name="points.json")
    # Named for a readable download; the name is the stable key (no content hash).
    assert url == "_assets/points.json"
    assert (tmp_path / url).read_bytes() == b'{"hello": "world"}'


def test_report_bundle_targets_export_dir(tmp_path: Path):
    from mini.reports import export_dir, export_key

    (tmp_path / "pyproject.toml").write_text("")
    nb = tmp_path / "docs" / "gpt-sweep" / "report.py"
    nb.parent.mkdir(parents=True)
    nb.write_text("")
    assert export_key(nb) == "gpt-sweep"  # a directory's report.py collapses to the dir
    assert export_dir(nb) == tmp_path / ".mini" / "exports" / "gpt-sweep"
    pub = report_bundle(nb)
    assert pub.asset_dir == tmp_path / ".mini" / "exports" / "gpt-sweep" / "_assets"
    assert pub.link == "_assets"


def test_export_key_top_level_notebook(tmp_path: Path):
    from mini.reports import export_key

    (tmp_path / "pyproject.toml").write_text("")
    nb = tmp_path / "docs" / "gpt.py"
    nb.parent.mkdir(parents=True)
    nb.write_text("")
    assert export_key(nb) == "gpt"  # top-level report → bare stem, served at /gpt/

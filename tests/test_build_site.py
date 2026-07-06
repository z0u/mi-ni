"""Tests for the static-site builder's author-link resolver (pure policy)."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_site", Path(__file__).resolve().parent.parent / "scripts" / "build_site.py"
)
assert _SPEC and _SPEC.loader
build_site = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_site)


@pytest.mark.parametrize(
    "url, want",
    [
        ("probe/report/index.html", "probe/report/"),
        ("probe/report/index.html#cell-3", "probe/report/#cell-3"),
        ("index.html", ""),
        ("../acts/report/index.html", "../acts/report/"),
        ("guide.html", "guide.html"),  # not an index page — untouched
        ("reindex.html", "reindex.html"),  # only a whole index.html segment is stripped
    ],
)
def test_strip_index(url, want):
    assert build_site._strip_index(url) == want


@pytest.fixture
def resolver() -> "build_site.LinkResolver":
    # Reports render to <key>/index.html (per-report dirs); markdown to <name>.html.
    return build_site.LinkResolver(
        render_map={
            "probe/report.py": "probe/report/index.html",
            "acts/report.py": "acts/report/index.html",
            "guide.md": "guide.html",
        },
        source_files=frozenset({"probe/experiment.py", "acts/experiment.py", "probe/report.py"}),
        site_base="https://o.github.io/r/",
        source_base="https://github.com/o/r/blob/main/",
    )


def test_nav_urls_absolute_when_externalizing(resolver):
    # With an asset <base>, the index link must be absolute (the site root); source is
    # always the notebook on GitHub.
    index, source = build_site._nav_urls(resolver, key="pipeline", nb_rel="docs/pipeline/report.py", externalizing=True)
    assert index == "https://o.github.io/r/"
    assert source == "https://github.com/o/r/blob/main/docs/pipeline/report.py"


def test_nav_urls_index_is_relative_when_localizing(resolver):
    # No <base> offline, so climb back to _site/index.html from _site/<key>/index.html.
    index, _ = build_site._nav_urls(resolver, key="pipeline", nb_rel="docs/pipeline/report.py", externalizing=False)
    assert index == "../index.html"
    index, _ = build_site._nav_urls(resolver, key="a/b", nb_rel="docs/a/b.py", externalizing=False)
    assert index == "../../index.html"


def test_rendered_link_is_absolute_pages_url_when_externalizing(resolver):
    # Published links drop index.html — GitHub Pages serves the directory form.
    got = resolver.resolve("../acts/report.py", from_dir="probe", out_dir="probe/report", externalizing=True)
    assert got == "https://o.github.io/r/acts/report/"


def test_rendered_link_stays_relative_when_localizing(resolver):
    # No <base> locally, so a relative link navigates within _site — and it's relative
    # to where *this* report renders (probe/report/), not its source dir (probe/).
    got = resolver.resolve("../acts/report.py", from_dir="probe", out_dir="probe/report", externalizing=False)
    assert got == "../../acts/report/index.html"


def test_source_file_resolves_to_github(resolver):
    got = resolver.resolve("./experiment.py", from_dir="probe", out_dir="probe/report", externalizing=True)
    assert got == "https://github.com/o/r/blob/main/docs/probe/experiment.py"


def test_fragment_is_preserved(resolver):
    got = resolver.resolve("../acts/report.py#cell-3", from_dir="probe", out_dir="probe/report", externalizing=True)
    assert got == "https://o.github.io/r/acts/report/#cell-3"


def test_repo_source_link_outside_docs_resolves_to_github(resolver):
    # A report linking to its source modules escapes docs/ but stays in the repo;
    # it should resolve to the GitHub source so it survives the asset <base>.
    # (Fixture has repo_root=None, so existence is trusted.)
    assert (
        resolver.resolve("../src/experiment", from_dir=".", out_dir=".", externalizing=True)
        == "https://github.com/o/r/blob/main/src/experiment"
    )
    assert (
        resolver.resolve(
            "../../src/experiment/model/README.md#gate",
            from_dir="gpt-sweep",
            out_dir="gpt-sweep/report",
            externalizing=True,
        )
        == "https://github.com/o/r/blob/main/src/experiment/model/README.md#gate"
    )


def test_link_escaping_the_repo_root_is_unresolved(resolver):
    assert resolver.resolve("../../../etc/passwd", from_dir="probe", out_dir="probe/report", externalizing=True) is None


def test_missing_repo_source_target_is_unresolved(tmp_path):
    # With a repo_root set, a link to a path that doesn't exist is left to warn.
    r = build_site.LinkResolver(
        render_map={},
        source_files=frozenset(),
        site_base=None,
        source_base="https://github.com/o/r/blob/main/",
        repo_root=tmp_path,
    )
    assert r.resolve("../src/nope", from_dir=".", out_dir=".", externalizing=True) is None
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.py").write_text("")
    assert (
        r.resolve("../src/real.py", from_dir=".", out_dir=".", externalizing=True)
        == "https://github.com/o/r/blob/main/src/real.py"
    )


def test_external_and_anchored_links_are_left_alone(resolver):
    kw = dict(from_dir="probe", out_dir="probe/report", externalizing=True)
    assert resolver.resolve("https://example.com", **kw) is None
    assert resolver.resolve("#section", **kw) is None
    assert resolver.resolve("/absolute", **kw) is None


def test_unknown_target_is_unresolved(resolver):
    assert resolver.resolve("./nope.py", from_dir="probe", out_dir="probe/report", externalizing=True) is None


def test_source_only_report_link_resolves_to_github():
    # A source-only example (e.g. gpt.py) is absent from render_map but still a file under
    # docs/, so a link to it (as from docs/index.md) falls through to the GitHub source
    # rather than a rendered page that would never exist. Markdown resolves in localize mode.
    r = build_site.LinkResolver(
        render_map={"pipeline/report.py": "pipeline/report/index.html"},
        source_files=frozenset({"gpt.py", "pipeline/report.py"}),
        site_base="https://o.github.io/r/",
        source_base="https://github.com/o/r/blob/main/",
    )
    assert (
        r.resolve("./gpt.py", from_dir="", out_dir="", externalizing=False)
        == "https://github.com/o/r/blob/main/docs/gpt.py"
    )


def test_missing_bases_degrade_to_unresolved():
    r = build_site.LinkResolver(
        render_map={"acts/report.py": "acts/report/index.html"},
        source_files=frozenset({"probe/experiment.py"}),
        site_base=None,
        source_base=None,
    )
    kw = dict(from_dir="probe", out_dir="probe/report")
    # Externalizing needs an absolute target; with no base it can't make one.
    assert r.resolve("../acts/report.py", externalizing=True, **kw) is None
    # …but localize still keeps rendered links relative (no base needed).
    assert r.resolve("../acts/report.py", externalizing=False, **kw) == "../../acts/report/index.html"

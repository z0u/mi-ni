"""Tests for the static-site builder's author-link resolver (pure policy)."""

import pytest

from mini.reports import github_slug

from tests.conftest import load_script

build_site = load_script("build_site")


@pytest.mark.parametrize(
    "url, want",
    [
        ("probe/report/index.html", "probe/report/"),
        ("probe/report/index.html#cell-3", "probe/report/#cell-3"),
        ("index.html", ""),
        ("reindex.html", "reindex.html"),  # not an index page: only a whole index.html segment is stripped
    ],
)
def test_strip_index(url, want):
    assert build_site._strip_index(url) == want


@pytest.mark.parametrize(
    "heading",
    [
        "Provenance & cost",  # the `&` goes, its two spaces both become hyphens
        "Fig 2.1: width in a transformer",
        "Hotfix safety: avoid double-spending",
        "`test_local_apparatus_concurrent` failed on a pristine tree",
        "Keeps_underscores",
    ],
)
def test_rendered_heading_ids_are_github_slugs(heading):
    """A `#fragment` is checked against GitHub's slugs, so the published page has to use them too.

    Python-Markdown's own slugify collapses a run of separators — it would render "Provenance & cost" as `provenance-cost` where GitHub and the check both say `provenance--cost`, and the link would resolve everywhere except the site it was written for.
    """
    html = build_site.render_markdown(f"## {heading}\n")

    assert f'id="{github_slug(heading)}"' in html


def test_a_heading_inside_a_details_block_is_given_an_id():
    """The index collapses its older sections, and a fragment into one has to land."""
    html = build_site.render_markdown(
        '<details markdown="1"><summary><h3>Iteration 0 (prep)</h3></summary>\n\nbody\n\n</details>\n'
    )

    assert 'id="iteration-0-prep"' in html


def test_a_mermaid_fence_becomes_the_element_the_library_renders_into():
    """Python-Markdown nests every fence in a `<code>`, which mermaid walks straight past."""
    html, has_mermaid = build_site.promote_mermaid(
        build_site.render_markdown('```mermaid\nflowchart LR\na(["suppress red"]) & b --> c\n```\n')
    )

    assert has_mermaid
    assert html == '<pre class="mermaid">flowchart LR\na([&quot;suppress red&quot;]) &amp; b --&gt; c\n</pre>'


def test_an_ordinary_fence_is_left_as_code():
    html, has_mermaid = build_site.promote_mermaid(build_site.render_markdown("```python\nx = 1\n```\n"))

    assert not has_mermaid
    assert 'class="mermaid"' not in html


@pytest.fixture
def resolver() -> "build_site.LinkResolver":
    # Reports render to <key>/index.html (per-report dirs); markdown to <name>.html.
    return build_site.LinkResolver(
        render_map={
            "probe/report.py": "probe/report/index.html",
            "acts/report.py": "acts/report/index.html",
            "acts/report": "acts/report/index.html",  # directory form: one report links another by its canonical URL
            "guide.md": "guide.html",
        },
        source_files=frozenset({"probe/experiment.py", "acts/experiment.py", "probe/report.py", "public/map.svg"}),
        site_base="https://o.github.io/r/",
        source_base="https://github.com/o/r/blob/main/",
        site_assets=frozenset({"public/map.svg"}),
    )


@pytest.fixture
def strips() -> dict[str, "build_site.FigureStrip"]:
    from mini.reports import ReportFigure

    return {
        "probe/report": build_site.FigureStrip(
            "probe/report",
            "https://hf.co/d/r/resolve/abc123/exports/probe/report/",
            (
                ReportFigure(
                    "grading",
                    light="_assets/grading-light.png",
                    dark="_assets/grading-dark.png",
                    alt="Bands",
                    width=640,
                    height=480,
                ),
                ReportFigure("extra", light="_assets/extra.png"),
            ),
        )
    }


def test_figures_marker_survives_markdown_and_expands_to_pinned_cdn_thumbnails(resolver, strips):
    """The marker rides inside a list item as a comment; expansion happens on the rendered HTML, after Python-Markdown (which would read a raw <div> indented in a list as code)."""
    body = build_site.render_markdown(
        "- [probe](./probe/report.py)\n\n    Lede.\n\n    <!-- mini:figures ./probe/report.py -->\n"
    )
    assert "mini:figures" in body  # the comment came through rendering intact

    out = build_site.expand_figure_strips(body, strips, resolver, from_dir="", externalizing=True)
    assert "mini:figures" not in out
    assert '<img src="https://hf.co/d/r/resolve/abc123/exports/probe/report/_assets/grading-light.png"' in out
    assert 'srcset="https://hf.co/d/r/resolve/abc123/exports/probe/report/_assets/grading-dark.png"' in out
    assert 'alt="Bands"' in out and 'loading="lazy"' in out
    assert 'width="640" height="480"' in out  # the export's stamped size, for layout before load
    # No anchor: a raw PNG opens transparent-on-white (wrong in dark mode); copying the
    # image in place gets the scheme-matched variant the <picture> shows.
    strip_html = out.split('<div class="fig-strip">')[1]
    assert "<a " not in strip_html
    assert strip_html.count("<picture>") == 1  # only the themed figure; the unthemed one is a bare <img>


def test_figures_marker_localizes_to_the_copied_assets(resolver, strips):
    out = build_site.expand_figure_strips(
        "<!-- mini:figures ./probe/report.py -->", strips, resolver, from_dir="", externalizing=False
    )
    assert '<img src="probe/report/_assets/grading-light.png"' in out  # beside _site/probe/report/index.html


def test_figures_marker_for_an_unbuilt_report_renders_nothing(resolver, strips, capsys):
    out = build_site.expand_figure_strips(
        "<!-- mini:figures ./acts/report.py --><p>after</p>", strips, resolver, from_dir="", externalizing=True
    )
    assert out == "<p>after</p>"
    assert "names no built report" in capsys.readouterr().out


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
    # Published links drop index.html — GitHub Pages serves the directory form — and a fragment rides along.
    got = resolver.resolve("../acts/report.py", from_dir="probe", out_dir="probe/report", externalizing=True)
    assert got == "https://o.github.io/r/acts/report/"
    got = resolver.resolve("../acts/report.py#cell-3", from_dir="probe", out_dir="probe/report", externalizing=True)
    assert got == "https://o.github.io/r/acts/report/#cell-3"


def test_rendered_link_stays_relative_when_localizing(resolver):
    # No <base> locally, so a relative link navigates within _site — and it's relative
    # to where *this* report renders (probe/report/), not its source dir (probe/).
    got = resolver.resolve("../acts/report.py", from_dir="probe", out_dir="probe/report", externalizing=False)
    assert got == "../../acts/report/index.html"


def test_directory_form_link_resolves_like_the_report_file(resolver):
    # A report links a sibling by its canonical published URL (``../acts/``, the directory),
    # not the notebook file — both must reach the same rendered page.
    assert (
        resolver.resolve("../acts/report/", from_dir="probe", out_dir="probe/report", externalizing=True)
        == "https://o.github.io/r/acts/report/"
    )
    assert (
        resolver.resolve("../acts/report/", from_dir="probe", out_dir="probe/report", externalizing=False)
        == "../../acts/report/index.html"
    )


def test_copied_asset_is_served_by_the_site_not_github(resolver):
    # An image copied into _site/ must point at the copy. A GitHub ``blob/`` URL is an
    # HTML page, so an <img> aimed at one renders nothing.
    assert resolver.resolve("./public/map.svg", from_dir="", out_dir="", externalizing=False) == "public/map.svg"
    assert (
        resolver.resolve("../public/map.svg", from_dir="probe", out_dir="probe/report", externalizing=False)
        == "../../public/map.svg"
    )


def test_copied_asset_is_absolute_when_externalizing(resolver):
    # Under an asset <base> a relative link would resolve against the bucket, so the
    # site root has to be spelled out.
    assert (
        resolver.resolve("../public/map.svg", from_dir="probe", out_dir="probe/report", externalizing=True)
        == "https://o.github.io/r/public/map.svg"
    )


def test_copy_assets_and_the_resolver_agree_on_what_lands_in_the_site(tmp_path, monkeypatch):
    # The two read one definition; this is the guard that they keep doing so.
    docs = tmp_path / "docs"
    (docs / "public").mkdir(parents=True)
    (docs / "public" / "map.svg").write_text("<svg/>")
    (docs / "index.md").write_text("# hi")
    (docs / "probe").mkdir()
    (docs / "probe" / "experiment.py").write_text("x = 1")
    (docs / "__marimo__").mkdir()
    (docs / "__marimo__" / "cache.json").write_text("{}")
    monkeypatch.setattr(build_site, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(build_site, "DOCS_DIR", docs)
    assert [p.relative_to(docs).as_posix() for p in build_site.site_asset_files()] == ["public/map.svg"]


def test_source_file_resolves_to_github(resolver):
    got = resolver.resolve("./experiment.py", from_dir="probe", out_dir="probe/report", externalizing=True)
    assert got == "https://github.com/o/r/blob/main/docs/probe/experiment.py"


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


def test_external_and_in_page_links_are_left_alone(resolver):
    kw = dict(from_dir="probe", out_dir="probe/report", externalizing=True)
    assert resolver.resolve("https://example.com", **kw) is None
    assert resolver.resolve("//cdn.example.com/x.js", **kw) is None
    assert resolver.resolve("#section", **kw) is None


def test_root_absolute_link_reads_against_the_repo_root(resolver):
    # The house style for a cross-tree link. One outside docs/ points at the GitHub
    # source; one under docs/ renders like the relative form of the same target.
    kw = dict(from_dir="probe", out_dir="probe/report", externalizing=True)
    assert resolver.resolve("/src/experiment", **kw) == "https://github.com/o/r/blob/main/src/experiment"
    assert resolver.resolve("/eng/gc.md#gate", **kw) == "https://github.com/o/r/blob/main/eng/gc.md#gate"
    assert resolver.resolve("/docs/acts/report.py", **kw) == "https://o.github.io/r/acts/report/"
    assert (
        resolver.resolve("/docs/probe/experiment.py", **kw)
        == "https://github.com/o/r/blob/main/docs/probe/experiment.py"
    )


def test_root_absolute_link_localizes_like_a_relative_one(resolver):
    got = resolver.resolve("/docs/acts/report.py", from_dir="probe", out_dir="probe/report", externalizing=False)

    assert got == resolver.resolve("../acts/report.py", from_dir="probe", out_dir="probe/report", externalizing=False)


def test_root_absolute_link_escaping_the_repo_root_is_unresolved(resolver):
    assert resolver.resolve("/../etc/passwd", from_dir="probe", out_dir="probe/report", externalizing=True) is None


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

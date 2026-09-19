"""Tests for the static-site builder's author-link resolver (pure policy)."""

import pytest
import json
from pathlib import Path

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
        "D2.1: anchoring in a transformer",
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

    out, has_strip = build_site.expand_figure_strips(body, strips, resolver, from_dir="", externalizing=True)
    assert has_strip and "mini:figures" not in out
    assert '<img src="https://hf.co/d/r/resolve/abc123/exports/probe/report/_assets/grading-light.png"' in out
    assert 'srcset="https://hf.co/d/r/resolve/abc123/exports/probe/report/_assets/grading-dark.png"' in out
    assert 'alt="Bands"' in out and 'loading="lazy"' in out
    assert 'width="640" height="480"' in out  # the export's stamped size, for layout before load
    # No anchor: a raw PNG opens transparent-on-white (wrong in dark mode) and navigates
    # off the index. The lightbox shows it in place instead, over a themed backdrop.
    strip_html = out.split('<div class="fig-strip">')[1]
    assert "<a " not in strip_html
    assert strip_html.count("<picture>") == 1  # only the themed figure; the unthemed one is a bare <img>


def test_figure_strip_shows_the_export_thumbnails_when_the_bundle_has_them(resolver):
    """A few KB per entry instead of the full figure; a bundle exported before thumbnails keeps serving the full-size image."""
    from mini.reports import ReportFigure

    strip = build_site.FigureStrip(
        "probe/report",
        "https://hf.co/d/r/resolve/abc123/exports/probe/report/",
        (
            ReportFigure(
                "grading",
                light="_assets/grading-light.png",
                dark="_assets/grading-dark.png",
                light_thumb="_assets/thumbs/grading-light.png",
                dark_thumb="_assets/thumbs/grading-dark.png",
                width=640,
                height=480,
            ),
            ReportFigure("old", light="_assets/old-light.png", dark="_assets/old-dark.png"),
        ),
    )
    out = build_site._figure_strip_html(strip, from_dir="", externalizing=True)
    base = "https://hf.co/d/r/resolve/abc123/exports/probe/report/"
    assert f'<img src="{base}_assets/thumbs/grading-light.png"' in out
    assert f'srcset="{base}_assets/thumbs/grading-dark.png"' in out
    assert 'width="640" height="480"' in out  # the figure's own aspect; the CSS fixes the height
    assert f'<img src="{base}_assets/old-light.png"' in out and f'srcset="{base}_assets/old-dark.png"' in out
    assert f'src="{base}_assets/grading-light.png"' not in out  # the strip never *loads* the full-size one


def test_figure_strip_thumbnails_name_the_full_size_figure_for_the_lightbox(resolver):
    """A strip thumbnail is unreadable at 4.5rem, so it carries the full-size URLs the overlay opens — both themes, since the index picks by device preference."""
    from mini.reports import ReportFigure

    base = "https://hf.co/d/r/resolve/abc123/exports/probe/report/"
    strip = build_site.FigureStrip(
        "probe/report",
        base,
        (
            ReportFigure(
                "grading",
                light="_assets/grading-light.png",
                dark="_assets/grading-dark.png",
                alt="Bands",
                light_thumb="_assets/thumbs/grading-light.png",
                dark_thumb="_assets/thumbs/grading-dark.png",
            ),
            ReportFigure("lone", light="_assets/lone.png", alt="One", light_thumb="_assets/thumbs/lone.png"),
        ),
    )
    out = build_site._figure_strip_html(strip, from_dir="", externalizing=True)
    assert f'data-mini-full="{base}_assets/grading-light.png"' in out
    assert f'data-mini-full-dark="{base}_assets/grading-dark.png"' in out
    assert out.count("data-mini-zoom") == 2  # both entries open, the unthemed one included
    assert "data-mini-full-dark" not in out.split("lone")[-1]  # nothing to offer for an unthemed figure
    assert out.count('tabindex="0"') == 2  # reachable without a mouse


def test_figure_strip_of_a_figureless_report_renders_nothing():
    """A report whose figures are inlined (no publisher) has no strip to show; an empty box would just add a margin."""
    strip = build_site.FigureStrip("probe/report", None, ())
    assert build_site._figure_strip_html(strip, from_dir="", externalizing=False) == ""


def test_figures_marker_localizes_to_the_copied_assets(resolver, strips):
    out, _ = build_site.expand_figure_strips(
        "<!-- mini:figures ./probe/report.py -->", strips, resolver, from_dir="", externalizing=False
    )
    assert '<img src="probe/report/_assets/grading-light.png"' in out  # beside _site/probe/report/index.html


def test_figures_marker_for_an_unbuilt_report_renders_nothing(resolver, strips, capsys):
    out, has_strip = build_site.expand_figure_strips(
        "<!-- mini:figures ./acts/report.py --><p>after</p>", strips, resolver, from_dir="", externalizing=True
    )
    assert out == "<p>after</p>" and not has_strip
    assert "names no built report" in capsys.readouterr().out


def test_nav_urls_absolute_when_externalizing(resolver):
    # With an asset <base>, the index link must be absolute (the site root); source is
    # always the report's source on GitHub.
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
    # rather than the report file — both must reach the same rendered page.
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
    (docs / "__pycache__").mkdir()
    (docs / "__pycache__" / "report.cpython-314.pyc").write_bytes(b"")
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


@pytest.mark.parametrize(
    ("base_href", "externalizing", "pdf", "expected"),
    [
        (
            "https://hf.co/d/r/resolve/abc123/exports/probe/report/",
            True,
            "https://z0u.github.io/mi-ni/probe/report/report.pdf",
            "https://z0u.github.io/mi-ni/probe/report/report.pdf",
        ),
        (None, False, "report.pdf", "probe/report/report.pdf"),
    ],
    ids=["externalize: absolute into the site, off the figures' CDN base", "localize: beside the copied page"],
)
def test_figure_strip_opens_with_the_pdf_the_build_printed(base_href, externalizing, pdf, expected):
    """The index links the same file the report's nav chip does."""
    from mini.reports import ReportFigure

    strip = build_site.FigureStrip("probe/report", base_href, (ReportFigure("f", light="_assets/f.png"),), pdf=pdf)
    out = build_site._figure_strip_html(strip, from_dir="", externalizing=externalizing)
    assert out.startswith(f'<div class="fig-strip"><a class="fig-strip-pdf" href="{expected}"')
    assert out.index("fig-strip-pdf") < out.index("<img")  # first, so it is never scrolled out of view


def test_figure_strip_of_a_figureless_report_still_links_its_pdf():
    strip = build_site.FigureStrip("probe/report", None, (), pdf="report.pdf")
    out = build_site._figure_strip_html(strip, from_dir="", externalizing=False)
    assert 'href="probe/report/report.pdf"' in out and "<img" not in out


def test_a_local_bundle_lists_every_rendition_the_page_declares(tmp_path: Path):
    """Localizing copies what the export wrote beside the page (a literate script's `index.md`), nothing the page declares but the export did not write, and never a PDF: an older export's is the build's to replace."""
    from mini.reports import MD_TYPE, PDF_TYPE, set_alternate

    (tmp_path / "pyproject.toml").write_text("")
    (nb := tmp_path / "docs" / "ex-1" / "report.py").parent.mkdir(parents=True)
    nb.write_text("")
    (bundle := tmp_path / ".mini" / "exports" / "ex-1").mkdir(parents=True)
    html = set_alternate("<html><head></head><body></body></html>", type=PDF_TYPE, href="report.pdf")
    html = set_alternate(html, type=MD_TYPE, href="index.md")
    (bundle / "index.html").write_text(html)
    (bundle / "index.md").write_text("# Hi")
    (bundle / "report.pdf").write_bytes(b"%PDF-stale")

    read = build_site._read_bundle(nb, store=None, pins={}, externalizing=False)

    assert read.renditions == (bundle / "index.md",)


class _Printer:
    """A stand-in for the print: writes a file naming the page, and remembers every call."""

    def __init__(self, works: bool = True):
        self.works = works
        self.calls: list[str] = []

    def __call__(self, serve_from: Path, out: Path, html: str) -> Path | None:
        assert serve_from.is_dir(), "the print serves the bundle from a directory"
        self.calls.append(html)
        if not self.works:
            return None
        out.write_bytes(f"%PDF of {html}".encode())
        return out


def test_the_pdf_memo_prints_a_page_once(tmp_path: Path):
    """The same page with the same tooling is the same PDF, so it is reused; a changed page, or a changed print, is printed again."""
    printer = _Printer()
    memo = build_site.PdfMemo(tmp_path, stamp="tooling-1", printer=printer)
    first = memo.pdf("probe", "<p>v1</p>", serve_from=tmp_path)
    assert first == tmp_path / "probe" / "report.pdf" and first.read_bytes() == b"%PDF of <p>v1</p>"
    memo.save()

    again = build_site.PdfMemo(tmp_path, stamp="tooling-1", printer=printer)
    assert again.pdf("probe", "<p>v1</p>", serve_from=tmp_path) == first
    assert printer.calls == ["<p>v1</p>"], "an unchanged page was printed again"

    again.pdf("probe", "<p>v2</p>", serve_from=tmp_path)
    build_site.PdfMemo(tmp_path, stamp="tooling-2", printer=printer).pdf("probe", "<p>v2</p>", serve_from=tmp_path)
    assert printer.calls == ["<p>v1</p>", "<p>v2</p>", "<p>v2</p>"]


def test_the_pdf_memo_forgets_a_report_the_build_no_longer_has(tmp_path: Path):
    memo = build_site.PdfMemo(tmp_path, stamp="t", printer=_Printer())
    memo.pdf("old", "<p>x</p>", serve_from=tmp_path)
    memo.save()
    memo = build_site.PdfMemo(tmp_path, stamp="t", printer=_Printer())
    memo.pdf("new", "<p>y</p>", serve_from=tmp_path)
    memo.save()
    assert json.loads((tmp_path / "pdfs.json").read_text()) == {"new": memo.key("<p>y</p>")}


def test_the_pdf_memo_records_nothing_when_nothing_can_print(tmp_path: Path):
    """No browser means no PDF and no manifest entry, so the next build with one prints it."""
    memo = build_site.PdfMemo(tmp_path, stamp="t", printer=_Printer(works=False))
    assert memo.pdf("probe", "<p>x</p>", serve_from=tmp_path) is None
    memo.save()
    assert json.loads((tmp_path / "pdfs.json").read_text()) == {}


def test_the_pdf_memo_reads_a_missing_or_broken_manifest_as_empty(tmp_path: Path):
    assert build_site.PdfMemo(tmp_path, stamp="t").previous == {}
    (tmp_path / "pdfs.json").write_text("{not json")
    assert build_site.PdfMemo(tmp_path, stamp="t").previous == {}


def test_the_printable_page_names_its_figures_on_the_cdn_and_keeps_its_fragments_bare():
    """The PDF's figures come off the pinned bundle, since the print serves the page alone; its `#footnote` links stay in-document, which a `<base>` would break."""
    links = build_site.LinkResolver(
        render_map={"probe/report.py": "probe/index.html"},
        source_files=frozenset(),
        site_base="https://z0u.github.io/mi-ni/",
        source_base="https://github.com/z0u/mi-ni/blob/main/docs/",
    )
    bundle = build_site._Bundle(
        '<html><head></head><body><img src="_assets/f.png"><a href="#fn1">1</a>'
        '<script>{"src":\\"_assets/g.png\\"}</script></body></html>',
        base_href="https://hf.co/d/r/resolve/abc/exports/probe/",
    )
    out = build_site._printable(bundle, links, from_dir="probe", key="probe", report_css="p{}")
    assert 'src="https://hf.co/d/r/resolve/abc/exports/probe/_assets/f.png"' in out
    assert '\\"https://hf.co/d/r/resolve/abc/exports/probe/_assets/g.png\\"' in out
    assert 'href="#fn1"' in out
    assert "<base" not in out and "p{}" in out

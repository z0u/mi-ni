import json
import re

import pytest

from mini.reports import (
    alternates,
    set_alternate,
    MANUAL_PUBLISH_MARKER,
    PROVENANCE_ASSET,
    PUBLISH_LOCK,
    SOURCE_ONLY_MARKER,
    Publisher,
    is_manually_published,
    export_key,
    externalize_html,
    input_dir,
    insert_base,
    is_report,
    lightbox_chrome,
    load_pins,
    mark_figures,
    relative_urls,
    report_figures,
    reports,
    rewrite_links,
    save_pins,
    set_banner,
    set_lightbox,
    set_provenance,
    set_report_styles,
    stray_links,
    write_thumbnails,
    use_publisher,
)

# Mimics an export: absolute CDN links + escaped data/asset URLs inside a JSON
# script blob, an author markdown link, and a relative asset reference.
SAMPLE = (
    "<!DOCTYPE html><html><head>"
    '<link rel="icon" href="https://cdn.jsdelivr.net/npm/x/favicon.ico" />'
    "</head><body>"
    '<script>{"cells":[{"outputs":[{"html":"<img src=\\"_assets/abc123.png\\" />'
    '<a href=\\"./experiment.py\\">src</a>'
    '<a href=\\"../acts/experiment.py\\">other</a>"}]}]}</script>'
    '<img src="data:image/png;base64,AAAA" />'
    '<a href="#section">jump</a>'
    "</body></html>"
)


def test_relative_urls_finds_only_relative():
    urls = set(relative_urls(SAMPLE))
    assert urls == {"_assets/abc123.png", "./experiment.py", "../acts/experiment.py"}
    # absolute, data:, and fragment URLs are excluded
    assert "https://cdn.jsdelivr.net/npm/x/favicon.ico" not in urls
    assert not any(u.startswith("data:") or u.startswith("#") for u in urls)


def test_report_figures_folds_themed_pairs_in_document_order():
    """The site index draws its thumbnails from the report's own HTML, which knows the narrative order."""
    html = (
        '<figure><img class="mini-themed-img-light" src="_assets/grading-light.png" alt="Three panels"'
        ' width="640" height="480" />'
        '<img class="mini-themed-img-dark" src="_assets/grading-dark.png" alt="Three panels"'
        ' width="641" height="480" /></figure>'
        '<img src="_assets/extra.png" alt="A lone diagram" />'
        '<img src="data:image/png;base64,AAAA" alt="inlined, not an asset" />'
    )
    figs = report_figures(html)
    assert [(f.stem, f.light, f.dark) for f in figs] == [
        ("grading", "_assets/grading-light.png", "_assets/grading-dark.png"),
        ("extra", "_assets/extra.png", None),
    ]
    assert figs[0].alt == "Three panels"
    assert (figs[0].width, figs[0].height) == (640, 480)  # the light tag's size, seen first
    assert figs[1].alt == "A lone diagram"
    assert (figs[1].width, figs[1].height) == (None, None)  # a plain tag carries none


def test_report_figures_reads_the_escaped_session_blob():
    """A script blob can bury a figure in JSON: \\u003C brackets, \\" quotes, escaped alt text."""
    html = (
        '<script>{"outputs":"\\u003Cimg class=\\"mini-themed-img-light\\" src=\\"_assets/cloud-light.png\\" '
        'alt=\\"Margin \\u2192 1; &quot;red&quot; holds\\" width=\\"512\\" height=\\"384\\" /\\u003E'
        '\\u003Cimg class=\\"mini-themed-img-dark\\" src=\\"_assets/cloud-dark.png\\" alt=\\"ditto\\" /\\u003E"}</script>'
    )
    figs = report_figures(html)
    assert [(f.stem, f.light, f.dark) for f in figs] == [
        ("cloud", "_assets/cloud-light.png", "_assets/cloud-dark.png"),
    ]
    # JSON-unescaped (\u2192 → the arrow), then HTML-unescaped (&quot; → "); the first alt seen wins.
    assert figs[0].alt == 'Margin → 1; "red" holds'
    assert (figs[0].width, figs[0].height) == (512, 384)  # read through the \" quoting


def test_report_figures_reads_thumbnails_from_the_bundle_meta():
    """An exported bundle declares its thumbnails in the one thing the build fetches: the HTML. Same leaf under the declared prefix, so no listing is needed; a bundle without the tag has none."""
    figs = '<img src="_assets/grading-light.png" alt="a" /><img src="_assets/grading-dark.png" alt="a" /><img src="_assets/extra.png" />'
    assert [(f.light_thumb, f.dark_thumb) for f in report_figures(figs)] == [(None, None), (None, None)]

    html = f'<head><meta name="mini-thumbnails" content="_assets/thumbs/" /></head>{figs}'
    assert [(f.light_thumb, f.dark_thumb) for f in report_figures(html)] == [
        ("_assets/thumbs/grading-light.png", "_assets/thumbs/grading-dark.png"),
        ("_assets/thumbs/extra.png", None),
    ]


@pytest.fixture
def bundle(tmp_path):
    """An exported bundle: a tall transparent figure pair, one already small, an SVG, and a dangling reference."""
    from PIL import Image

    assets = tmp_path / "_assets"
    assets.mkdir()
    big = Image.new("RGBA", (800, 600), (0, 0, 0, 0))
    big.paste((200, 30, 30, 255), (100, 100, 700, 500))  # an opaque block on a transparent ground
    big.save(assets / "cloud-light.png")
    Image.new("RGBA", (800, 600), (20, 20, 20, 255)).save(assets / "cloud-dark.png")  # opaque, like a themed figure
    Image.new("RGBA", (100, 50), (0, 0, 255, 255)).save(assets / "tiny.png")
    (assets / "diagram.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    html = (
        "<html><head><title>r</title></head><body>"
        '<img src="_assets/cloud-light.png" alt="c" /><img src="_assets/cloud-dark.png" alt="c" />'
        '<img src="_assets/tiny.png" /><img src="_assets/diagram.svg" /><img src="_assets/gone.png" />'
        "</body></html>"
    )
    return html, assets


def test_write_thumbnails_scales_declares_and_keeps_alpha(bundle):
    from PIL import Image

    html, assets = bundle
    out, written = write_thumbnails(html, assets, height=192)
    assert written == ["cloud-light.png", "cloud-dark.png", "tiny.png", "diagram.svg"]  # the dangling one is skipped
    assert '<head>\n    <meta name="mini-thumbnails" content="_assets/thumbs/" />' in out
    # The build reads the tag back into the same URLs the files were written at.
    assert [f.light_thumb for f in report_figures(out)][:2] == [
        "_assets/thumbs/cloud-light.png",
        "_assets/thumbs/tiny.png",
    ]

    with Image.open(assets / "thumbs" / "cloud-light.png") as thumb:
        assert thumb.size == (256, 192)  # 192 tall, aspect kept
        assert thumb.mode == "P"  # palette-quantized: a few KB rather than tens
        alpha = thumb.convert("RGBA").getchannel("A")
        assert alpha.getpixel((1, 1)) == 0  # the transparent ground survived quantization
        assert alpha.getpixel((128, 96)) in range(250, 256)  # the octree quantizer nudges alpha a little
    with Image.open(assets / "thumbs" / "cloud-dark.png") as thumb:
        assert thumb.mode == "P"
        assert thumb.convert("RGBA").getchannel("A").getextrema() == (255, 255)  # opaque stays exactly so
    assert (assets / "thumbs" / "cloud-light.png").stat().st_size < (assets / "cloud-light.png").stat().st_size


def test_write_thumbnails_copies_what_it_cannot_shrink(bundle):
    """The tag promises a thumbnail per figure, so an unscalable one is copied through rather than left to 404."""
    html, assets = bundle
    write_thumbnails(html, assets, height=192)
    for leaf in ("tiny.png", "diagram.svg"):  # already small; not a raster
        assert (assets / "thumbs" / leaf).read_bytes() == (assets / leaf).read_bytes()
    assert not (assets / "thumbs" / "gone.png").exists()


def test_write_thumbnails_is_deterministic_and_restamps(bundle):
    """A republish of an unchanged report must be a no-op commit: same bytes, and one tag rather than a stack of them."""
    html, assets = bundle
    once, _ = write_thumbnails(html, assets, height=192)
    first = (assets / "thumbs" / "cloud-light.png").read_bytes()
    twice, _ = write_thumbnails(once, assets, height=192)
    assert (assets / "thumbs" / "cloud-light.png").read_bytes() == first
    assert twice.count("mini-thumbnails") == 1


def test_mark_figures_defers_and_marks_asset_images_in_both_spellings():
    """A report ships every figure on load, both variants of a themed pair included. The mark reaches the tags a script blob buries as well as the ones written out as markup."""
    html = (
        '<img class="mini-themed-img-light" src="_assets/g-light.png" alt="A" width="640" height="480" />'
        '<img src="data:image/png;base64,AA" alt="inline" />'
        '<img src="https://example.com/off.png" alt="elsewhere" />'
        '<script>{"outputs":"\\u003Cimg src=\\"_assets/c-dark.png\\" alt=\\"x\\" /\\u003E"}</script>'
    )
    out = mark_figures(html)
    assert out.count("data-mini-zoom") == 2  # the two asset figures; not the inline or off-site ones
    assert '<img loading="lazy" tabindex="0" data-mini-zoom class="mini-themed-img-light"' in out
    # Inside the blob the tag's own quotes are escaped, so the added ones must be too —
    # otherwise the attribute closes the JSON string the tag lives in.
    assert '\\u003Cimg loading=\\"lazy\\" tabindex=\\"0\\" data-mini-zoom src=\\"_assets/c-dark.png\\"' in out
    blob = re.search(r"<script>(.*)</script>", out, re.S)
    assert blob and json.loads(blob.group(1))  # the tag still sits inside valid JSON


def test_mark_figures_leaves_an_already_marked_tag_alone():
    """Build steps re-run on their own output in a preview loop; a second pass must not stack attributes."""
    once = mark_figures('<img src="_assets/a.png" alt="a" />')
    assert mark_figures(once) == once


def test_set_lightbox_injects_one_overlay_before_the_styles_close():
    html = set_lightbox(_EXPORT_HTML)
    assert html.count("mini-lightbox") > 1 and html.index("mini-lightbox") < html.index("</head>")
    assert "showModal" in html  # a top-layer dialog, so no z-index race with the page
    assert "<a " not in lightbox_chrome()  # nothing that navigates away from the report


def test_lightbox_holds_the_box_open_while_the_full_size_figure_loads():
    js = lightbox_chrome()
    # The size the export stamped fixes the panel's shape *and* its width before any bytes
    # arrive, so it can't open flat or resize under the reader; the image on screen stands
    # in meanwhile, so the panel can't be left showing the figure opened before it.
    assert "aspectRatio" in js and "getAttribute('width')" in js
    assert "size(ar, w || " in js  # the physical size the report asked for is the ceiling
    assert "mini-lightbox-loading" in js  # the placeholder reads as one until the real image lands
    assert "big.src=full" in js and "pre.onload" in js  # swapped in only once decoded


def test_set_lightbox_is_a_noop_without_a_head():
    assert set_lightbox("<p>not a page</p>") == "<p>not a page</p>"


def test_stray_links_flags_author_links_not_assets():
    strays = stray_links(SAMPLE)
    assert strays == ["../acts/experiment.py", "./experiment.py"]  # sorted, deduped
    assert "_assets/abc123.png" not in strays  # the asset is allowed

    assets_only = '<img src="_assets/a.png"><img src=\\"_assets/b.png\\"><a href="https://x/y">x</a>'
    assert stray_links(assets_only) == []  # a page with nothing but assets is clean


def test_rewrite_links_handles_plain_and_escaped():
    # The author links from SAMPLE, mapped to absolute targets, must be replaced in
    # both their plain and JSON-escaped (\") forms; the asset ref is left alone.
    mapping = {
        "./experiment.py": "https://github.com/o/r/blob/main/docs/probe/experiment.py",
        "../acts/experiment.py": "https://github.com/o/r/blob/main/docs/acts/experiment.py",
    }
    out = rewrite_links(SAMPLE, mapping)
    assert '\\"https://github.com/o/r/blob/main/docs/probe/experiment.py\\"' in out
    assert '\\"https://github.com/o/r/blob/main/docs/acts/experiment.py\\"' in out
    assert 'experiment.py\\"' not in out.replace("docs/probe/experiment.py", "").replace(
        "docs/acts/experiment.py", ""
    )  # no original relative token survives
    assert "_assets/abc123.png" in out  # the asset reference is untouched


def test_rewrite_links_only_replaces_attribute_values():
    # A bare token sitting in text (not as a quoted attribute value) is left alone.
    html = 'see href="a/b.py" but the word a/b.py in prose stays'
    out = rewrite_links(html, {"a/b.py": "https://x/a/b.html"})
    assert 'href="https://x/a/b.html"' in out
    assert "the word a/b.py in prose stays" in out


def test_insert_base_pins_fragment_links_to_the_page():
    # A bare #fragment resolves against the <base>, i.e. to the bucket; with the page's URL
    # each one is spelled out as the same document, so footnotes and permalinks stay put.
    html = '<html><head></head><body><a href="#fn:1">1</a><a href=\'#top\'>t</a><a href="x.html#s">s</a></body></html>'
    out = insert_base(html, "https://cdn/b/", page_url="https://o.github.io/r/tour/")
    assert 'href="https://o.github.io/r/tour/#fn:1"' in out
    assert "href='https://o.github.io/r/tour/#top'" in out
    assert 'href="x.html#s"' in out  # a path with a fragment is the resolver's business, and stays
    assert insert_base(html, "https://cdn/b/").count("#fn:1") == 1  # no URL: left as written


def test_insert_base_adds_one_tag_in_head():
    out = insert_base("<html><head><meta></head><body></body></html>", "https://h/r/name/")
    assert out.count("<base ") == 1
    assert '<head>\n    <base href="https://h/r/name/" />' in out
    # base precedes the first resource so it governs it
    assert out.index("<base") < out.index("<meta")


def test_insert_base_only_first_head():
    # A literal "<head>" appearing later (e.g. in escaped content) is not touched.
    out = insert_base('<head></head><script>"\\u003chead\\u003e"</script>', "https://h/")
    assert out.count("<base ") == 1


def test_export_key_uses_docs_relative_stem(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    docs = tmp_path / "docs"
    (docs / "gpt-sweep").mkdir(parents=True)
    (docs / "gpt.py").write_text(_APP)
    (docs / "gpt-sweep" / "aside.py").write_text(_APP)
    assert export_key(docs / "gpt.py") == "gpt"
    assert export_key(docs / "gpt-sweep" / "aside.py") == "gpt-sweep/aside"


def test_export_key_drops_redundant_report_segment(tmp_path):
    # The canonical report of a directory publishes at the directory, not <dir>/report.
    (tmp_path / "pyproject.toml").write_text("")
    docs = tmp_path / "docs"
    (docs / "pipeline").mkdir(parents=True)
    (docs / "pipeline" / "report.py").write_text(_APP)
    assert export_key(docs / "pipeline" / "report.py") == "pipeline"
    # A top-level report.py has no directory to take, so it keeps its stem.
    (docs / "report.py").write_text(_APP)
    assert export_key(docs / "report.py") == "report"


def test_input_dir_is_the_report_own_directory(tmp_path):
    # A report that owns a directory reads the files beside it — experiment.py and friends.
    (tmp_path / "pyproject.toml").write_text("")
    docs = tmp_path / "docs"
    (docs / "pipeline").mkdir(parents=True)
    (docs / "pipeline" / "report.py").write_text(_APP)
    (docs / "pipeline" / "aside.py").write_text(_APP)
    assert input_dir(docs / "pipeline" / "report.py") == docs / "pipeline"
    assert input_dir(docs / "pipeline" / "aside.py") == docs / "pipeline"


def test_a_report_in_the_docs_root_has_no_input_dir(tmp_path):
    # The docs root is shared site space (publish.lock, index.md, report.css), so no
    # report may claim it: doing so would date every root-level report on every publish.
    (tmp_path / "pyproject.toml").write_text("")
    (docs := tmp_path / "docs").mkdir()
    (docs / "overview.py").write_text(_APP)
    assert input_dir(docs / "overview.py") is None


def test_pins_round_trip_sorted_and_diffable(tmp_path):
    (tmp_path / "docs").mkdir()
    assert load_pins(tmp_path) == {}  # no lock yet — nothing pinned
    save_pins(tmp_path, {"zeta": "b" * 40, "alpha": "a" * 40})
    assert load_pins(tmp_path) == {"alpha": "a" * 40, "zeta": "b" * 40}
    text = (tmp_path / PUBLISH_LOCK).read_text()
    assert text.index("alpha") < text.index("zeta")  # sorted → stable diffs, trivial merges
    assert text.endswith("\n")


def test_a_profile_keeps_its_pins_out_of_the_production_manifest(tmp_path, monkeypatch):
    """Under `MINI_PROFILE=dev` the pins go to a gitignored `.mini/` file; production's stays untouched and reachable."""
    from pathlib import Path

    from mini.reports import publish_lock

    (tmp_path / "docs").mkdir()
    save_pins(tmp_path, {"alpha": "a" * 40})  # a production pin, before any profile
    monkeypatch.setenv("MINI_PROFILE", "dev")
    assert publish_lock() == Path(".mini/publish.dev.lock")
    save_pins(tmp_path, {"alpha": "d" * 40})
    assert (tmp_path / ".mini" / "publish.dev.lock").exists()
    assert load_pins(tmp_path) == {"alpha": "d" * 40}  # the active manifest
    assert load_pins(tmp_path, profile=None) == {"alpha": "a" * 40}  # production, asked for by name


# A bare page shell: our bar is injected into it, not matched against existing markup.
_EXPORT_HTML = '<html><head><meta charset="utf-8" /></head><body><main class="lit"></main></body></html>'


def test_set_banner_injects_nav():
    out = set_banner(_EXPORT_HTML, index_url="https://o.github.io/r/", source_url="https://github.com/o/r/x.py")
    # Our bar is the first thing in <body>, so it paints above the report.
    assert out.index("<body>") < out.index("<nav data-mini-banner") < out.index('<main class="lit">')
    assert '<a href="https://o.github.io/r/" style=' in out and "&larr; Index" in out
    assert '<a href="https://github.com/o/r/x.py" style=' in out and ">Source</a>" in out
    # Absolute, not in-flow, so the chip floats above the page (and scrolls with it).
    assert "position:absolute" in out[out.index("<nav data-mini-banner") :][:200]
    # The content column is padded down so the report title isn't tucked under the chip.
    assert "main.lit{padding-top:3rem}" in out and out.index("padding-top:3rem") < out.index("</head>")


def test_set_banner_omits_missing_links():
    out = set_banner(_EXPORT_HTML, index_url="../index.html", source_url=None)
    assert "&larr; Index" in out
    assert ">Source<" not in out
    assert ">PDF<" not in out
    assert set_banner(_EXPORT_HTML) == _EXPORT_HTML  # no link at all: no bar at all


def test_set_banner_links_the_pdf_after_the_source():
    out = set_banner(_EXPORT_HTML, source_url="https://github.com/o/r/x.py", pdf_url="report.pdf")
    assert out.index(">Source</a>") < out.index('<a href="report.pdf" style=')
    assert ">PDF</a>" in out


def test_set_alternate_declares_a_rendition_in_the_head_and_restamps_by_type():
    out = set_alternate(_EXPORT_HTML, type="application/pdf", href="report.pdf")
    assert (
        out.index("<head>")
        < out.index('<link rel="alternate" type="application/pdf" href="report.pdf" />')
        < out.index("</head>")
    )
    out = set_alternate(out, type="text/markdown", href="report.md")
    out = set_alternate(out, type="application/pdf", href="print.pdf")  # a re-export replaces, never stacks
    assert alternates(out) == {"application/pdf": "print.pdf", "text/markdown": "report.md"}
    assert out.count('rel="alternate"') == 2


def test_stray_links_treats_a_declared_alternate_as_bundle_local():
    html = set_alternate(
        '<html><head></head><body><a href="./experiment.py">src</a><img src="_assets/f.png"></body></html>',
        type="application/pdf",
        href="report.pdf",
    )
    assert stray_links(html) == ["./experiment.py"]


_PRODUCER = {"experiment": "prep", "git_describe": "v1-3-gabc1234", "git_dirty": True, "run_at": "2026-07-12T01:02:03"}


def test_note_ref_maintains_the_provenance_sidecar(tmp_path):
    pub = Publisher(asset_dir=tmp_path / "_assets")
    pub.note_ref("shared/curves", _PRODUCER)
    pub.note_ref("shared/anon", None)  # read, but unattributable — still evidence
    sidecar = json.loads((tmp_path / "_assets" / PROVENANCE_ASSET).read_text())
    assert sidecar["refs"]["shared/curves"]["experiment"] == "prep"
    assert sidecar["refs"]["shared/anon"] is None
    before = (tmp_path / "_assets" / PROVENANCE_ASSET).read_text()
    pub.note_ref("shared/curves", _PRODUCER)  # re-resolving the same ref is a no-op rewrite
    assert (tmp_path / "_assets" / PROVENANCE_ASSET).read_text() == before


def test_get_ref_notes_into_the_active_publisher(tmp_path):
    from mini.store import LocalStore, producer_context

    store = LocalStore(tmp_path / "store")
    with producer_context({"experiment": "prep"}):
        store.set_ref("shared/a", store.put(b"a", name="a.bin"))
    pub = use_publisher(Publisher(asset_dir=tmp_path / "_assets"))
    try:
        store.get_ref("shared/a")
    finally:
        use_publisher(None)
    assert pub is not None
    sidecar = json.loads((tmp_path / "_assets" / PROVENANCE_ASSET).read_text())
    assert sidecar["refs"]["shared/a"]["experiment"] == "prep"


def test_asset_url_reserves_the_sidecar_name(tmp_path):
    pub = Publisher(asset_dir=tmp_path / "_assets")
    with pytest.raises(ValueError, match="reserved"):
        pub.asset_url(b"{}", name=PROVENANCE_ASSET)


def test_set_report_styles_inlines_the_sheet_last_in_head():
    css = ".sw { background: var(--sw) }"
    out = set_report_styles(_EXPORT_HTML, css)
    assert f"<style>\n{css}" in out  # inlined verbatim, not linked
    assert out.index(css) < out.index("</head>")  # lands inside <head>…
    # …and after any earlier <head> content, so it wins specificity ties with the page's own sheet.
    assert out.index('<meta charset="utf-8"') < out.index(css)


def test_set_report_styles_is_noop_without_css_or_head():
    assert set_report_styles(_EXPORT_HTML, "") == _EXPORT_HTML  # empty sheet: nothing to inline
    assert set_report_styles(_EXPORT_HTML, "   \n  ") == _EXPORT_HTML  # blank-only, too
    assert set_report_styles("<body>hi</body>", ".sw{}") == "<body>hi</body>"  # no </head> to hook


def test_set_provenance_injects_a_folded_footer():
    out = set_provenance(_EXPORT_HTML, {"shared/curves": _PRODUCER, "shared/other": {"experiment": "prep"}})
    assert out.index("<body>") < out.index("<details data-mini-provenance") < out.index('<main class="lit">')
    assert "<strong>prep</strong>" in out and "<code>v1-3-gabc1234</code> (dirty)" in out
    assert "run 2026-07-12" in out
    assert "via shared/curves, shared/other" in out  # both refs fold into one experiment entry
    assert "@media print{[data-mini-provenance]{display:none}}" in out  # hidden in print, like the banner
    # Absolute like the nav, so it floats above the page.
    assert "position:absolute" in out[out.index("<details data-mini-provenance") :][:200]


def test_set_provenance_is_noop_without_attributable_producers():
    assert set_provenance(_EXPORT_HTML, {}) == _EXPORT_HTML
    assert set_provenance(_EXPORT_HTML, {"shared/anon": None}) == _EXPORT_HTML


_APP = "# title: A report\n"


def test_reports_are_literate_scripts_by_their_header(tmp_path):
    """The report set is what the site renders: a literate script by its header. A plain module beside one is not a report, and the source-only marker opts either form out."""
    (tmp_path / "ex-1").mkdir()
    (nb := tmp_path / "ex-1" / "report.py").write_text(_APP)
    (tmp_path / "ex-2").mkdir()
    (lit := tmp_path / "ex-2" / "report.py").write_text('# title: Ex 2\n\n"""# Ex 2\n"""\nx = 1\n')
    (tmp_path / "ex-2" / "experiment.py").write_text("def main(ctx): ...\n")
    (tmp_path / "gpt.py").write_text(f'# title: A worked example\n# {SOURCE_ONLY_MARKER}\n"""prose"""\n')
    assert is_report(nb) and is_report(lit)
    assert not is_report(tmp_path / "ex-2" / "experiment.py")
    assert not is_report(tmp_path / "missing.py")
    found = {p.relative_to(tmp_path).as_posix() for p in reports(tmp_path)}
    assert found == {"ex-1/report.py", "ex-2/report.py"}


def test_manual_publish_marker_opts_out_of_the_reminder_only(tmp_path):
    plain = tmp_path / "report.py"
    plain.write_text(_APP)
    assert not is_manually_published(plain)  # reports are publish-checked by default

    nb = tmp_path / "manual.py"
    nb.write_text(f"# title: Manual\n# {MANUAL_PUBLISH_MARKER} — published on its own schedule\n")
    assert is_report(nb)  # still a report: rendered, pinned, on the site
    assert is_manually_published(nb)  # just not nagged about


def test_externalize_html_writes_sidecar_and_stamps_the_inline_copy(tmp_path):
    pub = Publisher(tmp_path / "_assets")
    html = '<div role="img"><svg xmlns="http://www.w3.org/2000/svg"></svg></div>'
    inline = externalize_html(html, name="sublines", publish=pub)
    # The inline copy differs by one inert attribute, on the root element and nowhere else…
    assert inline == '<div data-mini-asset="_assets/sublines.html" role="img">' + html.split(">", 1)[1]
    assert (tmp_path / "_assets" / "sublines.html").read_text() == html  # …and the file is the figure itself


def test_externalize_html_leaves_a_fragment_with_no_root_element_alone(tmp_path):
    # Nothing to hang the marker on, so the render can't swap it for a link; the sidecar
    # is still written, and the report still shows what it always did.
    pub = Publisher(tmp_path / "_assets")
    assert externalize_html("bare text", name="odd", publish=pub) == "bare text"
    assert (tmp_path / "_assets" / "odd.html").read_text() == "bare text"

    externalize_html("<svg xmlns='http://www.w3.org/2000/svg'/>", name="spark.svg", publish=pub)
    assert (tmp_path / "_assets" / "spark.svg").exists()  # an explicit extension is kept as given


def test_externalize_html_uses_the_default_publisher_when_there_is_one(tmp_path):
    use_publisher(None)
    assert externalize_html("<p>hi</p>", name="chunk") == "<p>hi</p>"  # nowhere to write: pass through

    use_publisher(Publisher(tmp_path / "_assets"))
    try:
        externalize_html("<p>hi</p>", name="chunk")
    finally:
        use_publisher(None)
    assert (tmp_path / "_assets" / "chunk.html").exists()


def test_body_injection_skips_a_body_tag_quoted_in_the_head():
    """report.css once said "theme on <body>" in a comment; the chips and the flash guard landed inside that <style>, unrendered."""
    from mini.reports import set_provenance

    html = (
        "<html><head><style>/* an explicit theme on <body> wins, where <body> states none */</style></head>"
        '<body class="x"><main class="lit"></main></body></html>'
    )
    for out in (
        set_banner(html, index_url="i/"),
        set_provenance(html, {"a": {"experiment": "x", "run": "r"}}),
    ):
        body = out[out.index('<body class="x">') :]
        assert out.index("</style>") < out.index("</head>") < out.index("<body class=")
        head = out[: out.index("</head>")]
        injected = ("<nav data-mini-banner", "<details data-mini-provenance", "<script>")
        assert any(tag in body for tag in injected), out
        assert not any(tag in head for tag in injected), out


def test_link_externalized_swaps_a_stamped_element_for_a_link(tmp_path, caplog):
    from mini.reports import link_externalized

    pub = Publisher(tmp_path / "_assets")
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><g><path d="M0 0"/></g></svg>'
    figure = externalize_html(
        f'<figure aria-label="A strip of &quot;marks&quot;">{svg}<figcaption>c</figcaption></figure>',
        name="strip",
        publish=pub,
    )
    md = f"before\n\n{figure}\n\nafter"
    assert (
        link_externalized(md) == 'before\n\n\n\n[A strip of "marks"](_assets/strip.html)\n\n\n\nafter'
    )  # nested tags, whole element gone

    image = externalize_html(svg, name="spark.svg", publish=pub)
    assert (
        link_externalized(image) == "\n\n![spark](_assets/spark.svg)\n\n"
    )  # an image sidecar is an image; no label → stem
    assert "spark.svg carries no aria-label" in caplog.text
    assert link_externalized("<p>plain</p>") == "<p>plain</p>"

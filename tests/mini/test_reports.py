import json

import pytest

from mini.reports import (
    MANUAL_PUBLISH_MARKER,
    PROVENANCE_ASSET,
    PUBLISH_LOCK,
    SOURCE_ONLY_MARKER,
    Publisher,
    is_manually_published,
    export_dir,
    export_key,
    externalize_html,
    input_dir,
    insert_base,
    is_report_notebook,
    load_pins,
    relative_urls,
    render_path,
    report_figures,
    report_notebooks,
    rewrite_links,
    save_pins,
    set_banner,
    set_provenance,
    set_report_styles,
    set_responsive,
    set_theme,
    stray_links,
    use_publisher,
)

# Mimics a Marimo export: absolute CDN links + escaped data/asset URLs inside the JSON
# session blob, an author markdown link, and a relative asset reference.
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
    """A Marimo export buries its figures in JSON: \\u003C brackets, \\" quotes, escaped alt text."""
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


# Mimics a Marimo export: the flat display block in the frozen mount config, plus the
# <head>/<body> the flicker guard hooks into.
_MOUNT_CONFIG = '<script>{"config": {"display": {"cell_output": "below", "theme": "light"}, "save": {}}}</script>'
_MOUNT = f'<html><head><meta charset="utf-8" /></head><body>{_MOUNT_CONFIG}<div id="root"></div></body></html>'


def test_set_theme_rewrites_display_theme():
    out = set_theme(_MOUNT)
    assert '"theme": "system"' in out
    assert '"theme": "light"' not in out
    # only the display theme changed; the rest of the config is intact
    assert '"cell_output": "below"' in out
    assert '"save": {}' in out


def test_set_theme_system_suppresses_flicker():
    out = set_theme(_MOUNT)
    # color-scheme meta (UA chrome) goes in <head>; the blocking guard (content) in <body>
    assert '<meta name="color-scheme" content="light dark" />' in out
    assert "prefers-color-scheme: dark" in out
    assert out.index('color-scheme" content') < out.index("</head>")
    assert out.index("<body>") < out.index("prefers-color-scheme")


def test_set_theme_fixed_target_skips_the_flash_guard():
    out = set_theme(_MOUNT.replace('"light"', '"dark"'), theme="dark")
    assert '"theme": "dark"' in out
    # a baked theme doesn't flash, so no blocking script — just declare the scheme
    assert '<meta name="color-scheme" content="dark" />' in out
    assert "prefers-color-scheme" not in out


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


def test_render_path_names_the_markdown_by_the_same_key(tmp_path):
    # The bundle and the Markdown render are two views of one report, so one key names both.
    (tmp_path / "pyproject.toml").write_text("")
    docs = tmp_path / "docs"
    (docs / "m2" / "ex-1").mkdir(parents=True)
    (docs / "m2" / "ex-1" / "report.py").write_text(_APP)
    nb = docs / "m2" / "ex-1" / "report.py"
    assert render_path(nb) == tmp_path / ".mini" / "renders" / "m2" / "ex-1.md"
    assert export_dir(nb) == tmp_path / ".mini" / "exports" / "m2" / "ex-1"


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


# Marimo renders its banner client-side, so the export only carries an empty shell; our
# bar is injected into that, not matched against existing banner markup.
_EXPORT_HTML = '<html><head><meta charset="utf-8" /></head><body><div id="root"></div></body></html>'


def test_set_banner_injects_nav_and_hides_marimo():
    out = set_banner(_EXPORT_HTML, index_url="https://o.github.io/r/", source_url="https://github.com/o/r/x.py")
    # Our bar is the first thing in <body>, so it paints above the report.
    assert out.index("<body>") < out.index("<nav data-mini-banner") < out.index('<div id="root">')
    assert '<a href="https://o.github.io/r/" style=' in out and "&larr; Index" in out
    assert '<a href="https://github.com/o/r/x.py" style=' in out and ">Source</a>" in out
    # Marimo's own (client-rendered) banner is hidden via a rule in <head>.
    assert '[data-testid="static-notebook-banner"]{display:none' in out
    assert out.index("static-notebook-banner") < out.index("</head>")
    # Absolute, not in-flow: Marimo's app is an opaque z-index layer that paints over an
    # in-flow sibling, so the chip must float above it (and it scrolls with the page).
    assert "position:absolute" in out[out.index("<nav data-mini-banner") :][:200]
    # The content column is padded down so the report title isn't tucked under the chip.
    assert '[class~="min-w-[400px]"]{padding-top:3rem}' in out


def test_set_banner_omits_missing_links():
    out = set_banner(_EXPORT_HTML, index_url="../index.html", source_url=None)
    assert "&larr; Index" in out
    assert ">Source<" not in out
    assert set_banner(_EXPORT_HTML) == _EXPORT_HTML  # neither link: no bar at all


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


def test_set_responsive_fits_narrow_screens_and_hides_watermark():
    out = set_responsive(_EXPORT_HTML)
    # The content column's 400px min-width is zeroed so it fits under ~400px…
    assert '[class~="min-w-[400px]"]{min-width:0!important}' in out
    # …and Marimo's bottom-right "made with marimo" watermark is hidden.
    assert '[data-testid="watermark"]{display:none!important}' in out
    assert out.index("min-w-[400px]") < out.index("</head>")  # both rules land in <head>


def test_set_report_styles_inlines_the_sheet_last_in_head():
    css = ".sw { background: var(--sw) }"
    out = set_report_styles(_EXPORT_HTML, css)
    assert f"<style>\n{css}" in out  # inlined verbatim, not linked
    assert out.index(css) < out.index("</head>")  # lands inside <head>…
    # …and after any earlier <head> content, so it wins specificity ties with Marimo's baked copy.
    assert out.index('<meta charset="utf-8"') < out.index(css)


def test_set_report_styles_is_noop_without_css_or_head():
    assert set_report_styles(_EXPORT_HTML, "") == _EXPORT_HTML  # empty sheet: nothing to inline
    assert set_report_styles(_EXPORT_HTML, "   \n  ") == _EXPORT_HTML  # blank-only, too
    assert set_report_styles("<body>hi</body>", ".sw{}") == "<body>hi</body>"  # no </head> to hook


def test_set_provenance_injects_a_folded_footer():
    out = set_provenance(_EXPORT_HTML, {"shared/curves": _PRODUCER, "shared/other": {"experiment": "prep"}})
    assert out.index("<body>") < out.index("<details data-mini-provenance") < out.index('<div id="root">')
    assert "<strong>prep</strong>" in out and "<code>v1-3-gabc1234</code> (dirty)" in out
    assert "run 2026-07-12" in out
    assert "via shared/curves, shared/other" in out  # both refs fold into one experiment entry
    assert "@media print{[data-mini-provenance]{display:none}}" in out  # hidden in print, like the banner
    # Absolute like the nav — floats above Marimo's opaque app layer instead of behind it.
    assert "position:absolute" in out[out.index("<details data-mini-provenance") :][:200]


def test_set_provenance_is_noop_without_attributable_producers():
    assert set_provenance(_EXPORT_HTML, {}) == _EXPORT_HTML
    assert set_provenance(_EXPORT_HTML, {"shared/anon": None}) == _EXPORT_HTML


_APP = "import marimo\napp = marimo.App()\n"


def test_is_report_notebook_detects_marimo_app(tmp_path):
    nb = tmp_path / "report.py"
    nb.write_text(_APP)
    assert is_report_notebook(nb)


def test_is_report_notebook_excludes_non_app_and_non_py(tmp_path):
    plain = tmp_path / "mod.py"
    plain.write_text("x = 1\n")
    assert not is_report_notebook(plain)
    assert not is_report_notebook(tmp_path / "notes.md")  # non-.py
    assert not is_report_notebook(tmp_path / "missing.py")  # absent


def test_report_notebooks_skips_source_only(tmp_path):
    (tmp_path / "report.py").write_text(_APP)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.py").write_text(_APP)
    example = tmp_path / "example.py"
    example.write_text(f"import marimo\n# {SOURCE_ONLY_MARKER} — heavy inline compute\napp = marimo.App()\n")
    (tmp_path / "plain.py").write_text("x = 1\n")
    assert not is_report_notebook(example)  # the marker is what takes it out
    found = {p.relative_to(tmp_path).as_posix() for p in report_notebooks(tmp_path)}
    assert found == {"report.py", "sub/nested.py"}


def test_manual_publish_marker_opts_out_of_the_reminder_only(tmp_path):
    plain = tmp_path / "report.py"
    plain.write_text(_APP)
    assert not is_manually_published(plain)  # reports are publish-checked by default

    nb = tmp_path / "manual.py"
    nb.write_text(f"import marimo\n# {MANUAL_PUBLISH_MARKER} — published on its own schedule\napp = marimo.App()\n")
    assert is_report_notebook(nb)  # still a report: rendered, pinned, on the site
    assert is_manually_published(nb)  # just not nagged about


def test_externalize_html_writes_sidecar_and_passes_through(tmp_path):
    pub = Publisher(tmp_path / "_assets")
    html = '<div role="img"><svg xmlns="http://www.w3.org/2000/svg"></svg></div>'
    assert externalize_html(html, name="sublines", publish=pub) == html  # inline copy unchanged
    assert (tmp_path / "_assets" / "sublines.html").read_text() == html  # …and a plain file for tooling

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


def test_virtualize_falls_back_to_the_file_url_off_the_kernel(tmp_path):
    # No marimo kernel under pytest, so there is nothing to register the bytes with and
    # nothing to serve them: the publisher must degrade to the file it just wrote rather
    # than to the data: URI mo.image would hand back.
    from mini.reports import _virtual_url

    pub = Publisher(asset_dir=tmp_path / "a", link="public/.mini/r", versioned=True, virtualize=True)
    url = pub.asset_url(b"png-bytes", name="fig.png")
    assert url.startswith("public/.mini/r/fig.png?v=")
    assert (tmp_path / "a" / "fig.png").read_bytes() == b"png-bytes"
    assert _virtual_url(tmp_path / "a" / "fig.png") is None  # the condition the fallback keys off


def test_virtualize_prefers_the_kernel_url_and_still_writes_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr("mini.reports._virtual_url", lambda p: f"./@file/9-{p.name}")
    pub = Publisher(asset_dir=tmp_path / "a", link="public/.mini/r", versioned=True, virtualize=True)
    url = pub.asset_url(b"png-bytes", name="fig.png")
    # The kernel mints a fresh name per render, so the ?v= cache-buster has nothing to do.
    assert url == "./@file/9-fig.png"
    # The readable copy stays on disk — it is what marimo reads, and what a person browsing
    # the directory finds.
    assert (tmp_path / "a" / "fig.png").read_bytes() == b"png-bytes"


def test_files_nothing_fetches_skip_the_kernel(tmp_path, monkeypatch):
    # An export sidecar is written for tooling to read off disk; nothing fetches it, so it
    # should not occupy a slot in the kernel's registry — whether the caller asks for that
    # with serve=False or reaches it through externalize_html.
    monkeypatch.setattr("mini.reports._virtual_url", lambda p: pytest.fail(f"{p.name} was virtualized"))
    pub = Publisher(asset_dir=tmp_path / "a", link="_assets", virtualize=True)
    assert pub.asset_url(b"<svg/>", name="frag.svg", serve=False) == "_assets/frag.svg"

    sidecar = Publisher(asset_dir=tmp_path / "b", virtualize=True)
    assert externalize_html("<svg/>", name="frag.svg", publish=sidecar) == "<svg/>"
    assert (tmp_path / "b" / "frag.svg").read_text() == "<svg/>"


def test_virtual_url_lifts_the_src_from_marimos_own_img_tag():
    # Pin the parse to marimo's HTML builder rather than to a hand-written sample, so a
    # change in how it quotes attributes shows up here.
    from marimo._output.builder import h

    from mini.reports import _IMG_SRC

    tag = h.img(src="./@file/49361-164816-TeHGmd7R.png", alt="a plot", style="max-width: 100%")
    m = _IMG_SRC.search(tag)
    assert m is not None, tag
    assert m.group(2) == "./@file/49361-164816-TeHGmd7R.png"


def test_report_bundle_virtualizes_only_interactively(tmp_path, monkeypatch):
    from mini.reports import EXPORTING_ENV, report_bundle

    nb = tmp_path / "docs" / "m9" / "report.py"
    nb.parent.mkdir(parents=True)
    nb.write_text("import marimo\n")
    monkeypatch.delenv(EXPORTING_ENV, raising=False)
    assert report_bundle(nb).virtualize is True
    monkeypatch.setenv(EXPORTING_ENV, "1")
    assert report_bundle(nb).virtualize is False

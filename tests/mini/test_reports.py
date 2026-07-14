import json

import pytest

from mini.reports import (
    PROVENANCE_ASSET,
    SOURCE_ONLY_MARKER,
    Publisher,
    export_key,
    insert_base,
    is_report_notebook,
    relative_urls,
    report_notebooks,
    rewrite_links,
    set_banner,
    set_provenance,
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


def test_stray_links_flags_author_links_not_assets():
    strays = stray_links(SAMPLE)
    assert strays == ["../acts/experiment.py", "./experiment.py"]  # sorted, deduped
    assert "_assets/abc123.png" not in strays  # the asset is allowed


def test_stray_links_empty_when_only_assets():
    html = '<img src="_assets/a.png"><img src=\\"_assets/b.png\\"><a href="https://x/y">x</a>'
    assert stray_links(html) == []


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


def test_set_theme_is_noop_without_a_theme():
    html = "<html><head></head><body></body></html>"
    assert set_theme(html) == html


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


def test_set_banner_omits_missing_links():
    out = set_banner(_EXPORT_HTML, index_url="../index.html", source_url=None)
    assert "&larr; Index" in out
    assert ">Source<" not in out


def test_set_banner_is_noop_without_urls():
    assert set_banner(_EXPORT_HTML) == _EXPORT_HTML


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


def test_set_provenance_injects_a_folded_footer():
    out = set_provenance(_EXPORT_HTML, {"shared/curves": _PRODUCER, "shared/other": {"experiment": "prep"}})
    assert out.index("<body>") < out.index("<details data-mini-provenance") < out.index('<div id="root">')
    assert "<strong>prep</strong>" in out and "<code>v1-3-gabc1234</code> (dirty)" in out
    assert "run 2026-07-12" in out
    assert "via shared/curves, shared/other" in out  # both refs fold into one experiment entry
    assert "@media print{[data-mini-provenance]{display:none}}" in out  # hidden in print, like the banner


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


def test_source_only_marker_opts_out(tmp_path):
    nb = tmp_path / "example.py"
    nb.write_text(f"import marimo\n# {SOURCE_ONLY_MARKER} — heavy inline compute\napp = marimo.App()\n")
    assert not is_report_notebook(nb)


def test_report_notebooks_skips_source_only(tmp_path):
    (tmp_path / "report.py").write_text(_APP)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.py").write_text(_APP)
    (tmp_path / "example.py").write_text(f"# {SOURCE_ONLY_MARKER}\n{_APP}")
    (tmp_path / "plain.py").write_text("x = 1\n")
    found = {p.relative_to(tmp_path).as_posix() for p in report_notebooks(tmp_path)}
    assert found == {"report.py", "sub/nested.py"}

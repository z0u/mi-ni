from mini.reports import (
    SOURCE_ONLY_MARKER,
    insert_base,
    is_report_notebook,
    relative_urls,
    report_notebooks,
    rewrite_links,
    set_theme,
    stray_links,
)

# Mimics a Marimo export: absolute CDN links + escaped data/asset URLs inside the JSON
# session blob, an author markdown link, and a relative asset reference.
SAMPLE = (
    '<!DOCTYPE html><html><head>'
    '<link rel="icon" href="https://cdn.jsdelivr.net/npm/x/favicon.ico" />'
    '</head><body>'
    '<script>{"cells":[{"outputs":[{"html":"<img src=\\"_assets/abc123.png\\" />'
    '<a href=\\"./experiment.py\\">src</a>'
    '<a href=\\"../acts/experiment.py\\">other</a>"}]}]}</script>'
    '<img src="data:image/png;base64,AAAA" />'
    '<a href="#section">jump</a>'
    '</body></html>'
)


def test_relative_urls_finds_only_relative():
    urls = set(relative_urls(SAMPLE))
    assert urls == {'_assets/abc123.png', './experiment.py', '../acts/experiment.py'}
    # absolute, data:, and fragment URLs are excluded
    assert 'https://cdn.jsdelivr.net/npm/x/favicon.ico' not in urls
    assert not any(u.startswith('data:') or u.startswith('#') for u in urls)


def test_stray_links_flags_author_links_not_assets():
    strays = stray_links(SAMPLE)
    assert strays == ['../acts/experiment.py', './experiment.py']  # sorted, deduped
    assert '_assets/abc123.png' not in strays  # the asset is allowed


def test_stray_links_empty_when_only_assets():
    html = '<img src="_assets/a.png"><img src=\\"_assets/b.png\\"><a href="https://x/y">x</a>'
    assert stray_links(html) == []


def test_rewrite_links_handles_plain_and_escaped():
    # The author links from SAMPLE, mapped to absolute targets, must be replaced in
    # both their plain and JSON-escaped (\") forms; the asset ref is left alone.
    mapping = {
        './experiment.py': 'https://github.com/o/r/blob/main/docs/probe/experiment.py',
        '../acts/experiment.py': 'https://github.com/o/r/blob/main/docs/acts/experiment.py',
    }
    out = rewrite_links(SAMPLE, mapping)
    assert '\\"https://github.com/o/r/blob/main/docs/probe/experiment.py\\"' in out
    assert '\\"https://github.com/o/r/blob/main/docs/acts/experiment.py\\"' in out
    assert 'experiment.py\\"' not in out.replace('docs/probe/experiment.py', '').replace(
        'docs/acts/experiment.py', ''
    )  # no original relative token survives
    assert '_assets/abc123.png' in out  # the asset reference is untouched


def test_rewrite_links_only_replaces_attribute_values():
    # A bare token sitting in text (not as a quoted attribute value) is left alone.
    html = 'see href="a/b.py" but the word a/b.py in prose stays'
    out = rewrite_links(html, {'a/b.py': 'https://x/a/b.html'})
    assert 'href="https://x/a/b.html"' in out
    assert 'the word a/b.py in prose stays' in out


def test_insert_base_adds_one_tag_in_head():
    out = insert_base('<html><head><meta></head><body></body></html>', 'https://h/r/name/')
    assert out.count('<base ') == 1
    assert '<head>\n    <base href="https://h/r/name/" />' in out
    # base precedes the first resource so it governs it
    assert out.index('<base') < out.index('<meta')


def test_insert_base_only_first_head():
    # A literal "<head>" appearing later (e.g. in escaped content) is not touched.
    out = insert_base('<head></head><script>"\\u003chead\\u003e"</script>', 'https://h/')
    assert out.count('<base ') == 1


# Mimics the flat display block in Marimo's frozen mount config.
_MOUNT = '<script>{"config": {"display": {"cell_output": "below", "theme": "light"}, "save": {}}}</script>'


def test_set_theme_rewrites_display_theme():
    out = set_theme(_MOUNT)
    assert '"theme": "system"' in out
    assert '"theme": "light"' not in out
    # only the display theme changed; the rest of the config is intact
    assert '"cell_output": "below"' in out
    assert '"save": {}' in out


def test_set_theme_accepts_existing_dark_and_custom_target():
    assert '"theme": "dark"' in set_theme(_MOUNT.replace('"light"', '"dark"'), theme='dark')


def test_set_theme_is_noop_without_a_theme():
    html = '<html><head></head><body></body></html>'
    assert set_theme(html) == html


_APP = 'import marimo\napp = marimo.App()\n'


def test_is_report_notebook_detects_marimo_app(tmp_path):
    nb = tmp_path / 'report.py'
    nb.write_text(_APP)
    assert is_report_notebook(nb)


def test_is_report_notebook_excludes_non_app_and_non_py(tmp_path):
    plain = tmp_path / 'mod.py'
    plain.write_text('x = 1\n')
    assert not is_report_notebook(plain)
    assert not is_report_notebook(tmp_path / 'notes.md')  # non-.py
    assert not is_report_notebook(tmp_path / 'missing.py')  # absent


def test_source_only_marker_opts_out(tmp_path):
    nb = tmp_path / 'example.py'
    nb.write_text(f'import marimo\n# {SOURCE_ONLY_MARKER} — heavy inline compute\napp = marimo.App()\n')
    assert not is_report_notebook(nb)


def test_report_notebooks_skips_source_only(tmp_path):
    (tmp_path / 'report.py').write_text(_APP)
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'nested.py').write_text(_APP)
    (tmp_path / 'example.py').write_text(f'# {SOURCE_ONLY_MARKER}\n{_APP}')
    (tmp_path / 'plain.py').write_text('x = 1\n')
    found = {p.relative_to(tmp_path).as_posix() for p in report_notebooks(tmp_path)}
    assert found == {'report.py', 'sub/nested.py'}

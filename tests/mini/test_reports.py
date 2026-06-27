from mini.reports import insert_base, relative_urls, rewrite_links, stray_links

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

#!/usr/bin/env python
"""Build the static site from Marimo HTML output."""

import re
import shutil
from pathlib import Path

import markdown as md_lib

from mini.reports import insert_base, stray_links

WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
SITE_DIR = WORKSPACE_ROOT / '_site'
DOCS_DIR = WORKSPACE_ROOT / 'docs'

# The relative dir, beside each report's HTML, holding its externalized assets
# (figures, data blobs) written by mini.vis.Publisher.
ASSET_LINK = '_assets'


def prepare_dirs():
    print('Preparing site directory...')
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()


def _resolve_store():
    """The project store — an HF bucket if configured + authed, else local.

    Drives the two asset modes: a bucket means *externalize* (upload + ``<base>``); a
    local store means *localize* (copy assets beside the HTML in ``_site``).
    """
    from mini.store import default_store

    return default_store(WORKSPACE_ROOT / '.mini' / 'store')


def _bundle_key(rel_parent: Path) -> str:
    """A bucket namespace for a report's assets, from its dir relative to docs/."""
    posix = rel_parent.as_posix()
    return posix if posix != '.' else '_root'


def _externalize_assets(assets_dir: Path, store, key: str) -> str:
    """Upload a bundle's assets to the bucket; return the ``<base href>`` they sit under.

    The relative ``_assets/<sha>.ext`` references then resolve against this base to the
    published bucket URL — content-addressed, so ``put`` is a no-op when bytes recur.
    """
    for f in sorted(p for p in assets_dir.iterdir() if p.is_file()):
        store.publish(store.put(f, name=f.name), f'reports/{key}/{ASSET_LINK}/{f.name}')
    return f'https://huggingface.co/buckets/{store.bucket}/resolve/published/reports/{key}/'


def copy_marimo_output():
    """Copy each ``__marimo__`` report's HTML to ``_site/``, resolving its asset bundle.

    Local store → copy ``_assets/`` beside the HTML (relative URLs resolve in ``_site``).
    HF bucket   → upload ``_assets/`` and insert one ``<base>`` so the same relative
    URLs resolve at the bucket; ``_site`` then carries only HTML.
    """
    print('Copying Marimo HTML output...')
    from mini.hf_store import HFStore

    store = _resolve_store()
    externalizing = isinstance(store, HFStore)
    print(f'  asset mode: {"externalize → " + store.bucket if externalizing else "localize (no bucket)"}')

    for marimo_dir in sorted(DOCS_DIR.rglob('__marimo__')):
        if not marimo_dir.is_dir():
            continue
        rel_parent = marimo_dir.parent.relative_to(DOCS_DIR)
        assets_dir = marimo_dir / ASSET_LINK
        has_assets = assets_dir.is_dir() and any(assets_dir.iterdir())

        base_href = None
        if has_assets and externalizing:
            base_href = _externalize_assets(assets_dir, store, _bundle_key(rel_parent))

        for html_file in sorted(marimo_dir.rglob('*.html')):
            rel = html_file.relative_to(marimo_dir)
            if ASSET_LINK in rel.parts:
                continue
            dest = SITE_DIR / rel_parent / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            html = html_file.read_text('utf-8')
            for stray in stray_links(html, link=ASSET_LINK):
                print(f'  ! {rel}: relative link {stray!r} — make it absolute, a <base> would break it')
            if base_href:
                html = insert_base(html, base_href)
            dest.write_text(html, 'utf-8')
            print(
                f'  {html_file.relative_to(WORKSPACE_ROOT)} -> {dest.relative_to(WORKSPACE_ROOT)}{" [+base]" if base_href else ""}'
            )

        if has_assets and not externalizing:
            out = SITE_DIR / rel_parent / ASSET_LINK
            shutil.copytree(assets_dir, out, dirs_exist_ok=True)
            print(f'  {assets_dir.relative_to(WORKSPACE_ROOT)}/ -> {out.relative_to(WORKSPACE_ROOT)}/')


def copy_assets():
    """Copy non-notebook, non-markdown files from docs/ to _site/."""
    print('Copying assets...')
    skip_dirs = {'__marimo__'}
    skip_suffixes = {'.py', '.md', '.ipynb'}
    for item in sorted(DOCS_DIR.rglob('*')):
        if not item.is_file():
            continue
        parts = item.relative_to(DOCS_DIR).parts
        if any(p in skip_dirs or p.startswith('.') for p in parts):
            continue
        if item.suffix in skip_suffixes:
            continue
        rel = item.relative_to(DOCS_DIR)
        dest = SITE_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f'  {item.relative_to(WORKSPACE_ROOT)} -> {dest.relative_to(WORKSPACE_ROOT)}')
        shutil.copy2(item, dest)


def site_root(dest: Path) -> str:
    """Return the relative path prefix from dest back to the site root."""
    depth = len(dest.relative_to(SITE_DIR).parts) - 1
    return '../' * depth


def copy_md_stylesheet():
    """Copy the Markdown page stylesheet to _site/."""
    print('Copying Markdown stylesheet...')
    css_src = WORKSPACE_ROOT / 'scripts' / 'md.css'
    css_dest = SITE_DIR / 'md.css'
    shutil.copy2(css_src, css_dest)
    print(f'  {css_src.relative_to(WORKSPACE_ROOT)} -> {css_dest.relative_to(WORKSPACE_ROOT)}')


def convert_markdown():
    """Convert all .md files in docs/ (except README.md) to .html in _site/."""
    print('Converting Markdown...')
    skip = {'README.md'}
    for md_file in sorted(DOCS_DIR.rglob('*.md')):
        if md_file.name in skip:
            continue
        rel = md_file.relative_to(DOCS_DIR).with_suffix('.html')
        dest = SITE_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = md_file.read_text('utf-8')
        text = re.sub(r'\]\(([^)]+)\.py\)', r'](\1.html)', text)
        body = md_lib.markdown(text, extensions=['extra'])
        title_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else md_file.stem
        root = site_root(dest)
        html = (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '<head>\n'
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{title}</title>\n'
            f'<link rel="stylesheet" href="{root}md.css">\n'
            '</head>\n'
            '<body>\n' + body + '\n</body>\n</html>\n'
        )
        dest.write_text(html, 'utf-8')
        print(f'  {md_file.relative_to(WORKSPACE_ROOT)} -> {dest.relative_to(WORKSPACE_ROOT)}')


def add_nojekyll():
    (SITE_DIR / '.nojekyll').touch()


def main():
    prepare_dirs()
    copy_marimo_output()
    copy_assets()
    copy_md_stylesheet()
    convert_markdown()
    add_nojekyll()
    print(f'\nSite written to {SITE_DIR.relative_to(WORKSPACE_ROOT)}/')


if __name__ == '__main__':
    main()

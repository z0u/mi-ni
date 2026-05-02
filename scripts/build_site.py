#!/usr/bin/env python
"""Build the static site from Marimo HTML output."""

import re
import shutil
from pathlib import Path

import markdown as md_lib

WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
SITE_DIR = WORKSPACE_ROOT / '_site'
DOCS_DIR = WORKSPACE_ROOT / 'docs'


def prepare_dirs():
    print('Preparing site directory...')
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()


def copy_marimo_output():
    """Copy HTML from all __marimo__ dirs in docs/ to _site/, preserving relative structure."""
    print('Copying Marimo HTML output...')
    for marimo_dir in sorted(DOCS_DIR.rglob('__marimo__')):
        if not marimo_dir.is_dir():
            continue
        rel_parent = marimo_dir.parent.relative_to(DOCS_DIR)
        for html_file in sorted(marimo_dir.rglob('*.html')):
            rel = html_file.relative_to(marimo_dir)
            dest = SITE_DIR / rel_parent / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f'  {html_file.relative_to(WORKSPACE_ROOT)} -> {dest.relative_to(WORKSPACE_ROOT)}')
            shutil.copy2(html_file, dest)


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


def convert_index():
    """Convert docs/index.md to _site/index.html, rewriting .py links to .html."""
    index_md = DOCS_DIR / 'index.md'
    if not index_md.exists():
        return
    print('Converting index...')
    text = index_md.read_text('utf-8')
    text = re.sub(r'\]\(([^)]+)\.py\)', r'](\1.html)', text)
    body = md_lib.markdown(text, extensions=['extra'])
    html = (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>mi-ni</title>\n'
        '</head>\n'
        '<body>\n'
        + body
        + '\n</body>\n</html>\n'
    )
    dest = SITE_DIR / 'index.html'
    dest.write_text(html, 'utf-8')
    print(f'  {index_md.relative_to(WORKSPACE_ROOT)} -> {dest.relative_to(WORKSPACE_ROOT)}')


def add_nojekyll():
    (SITE_DIR / '.nojekyll').touch()


def main():
    prepare_dirs()
    copy_marimo_output()
    copy_assets()
    convert_index()
    add_nojekyll()
    print(f'\nSite written to {SITE_DIR.relative_to(WORKSPACE_ROOT)}/')


if __name__ == '__main__':
    main()

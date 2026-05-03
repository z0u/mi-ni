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

#!/usr/bin/env python
"""Build the static site from Marimo HTML output."""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import markdown as md_lib

from mini.reports import insert_base, rewrite_links, stray_links

WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
SITE_DIR = WORKSPACE_ROOT / '_site'
DOCS_DIR = WORKSPACE_ROOT / 'docs'

# The relative dir, beside each report's HTML, holding its externalized assets
# (figures, data blobs) written by mini.reports.Publisher.
ASSET_LINK = '_assets'

# Source suffixes that the build renders into a sibling .html page (so an author link
# to one resolves to the rendered result, not the dead source file).
_RENDERED_SUFFIXES = ('.py', '.ipynb', '.md')


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

    The relative ``_assets/<sha>/<name>`` references then resolve against this base to
    the published bucket URL — content-addressed, so ``put`` is a no-op when bytes
    recur. Walks recursively so the ``<sha>/<name>`` layout is preserved under the key.

    Idempotent and read-only-safe: an already-published asset path is skipped (no
    ``put``/``publish`` write), so the CI Pages build — which holds only a read-only
    token and relies on the agent having published the assets at export time — just
    derives the ``<base>`` without writing to the bucket.
    """
    for f in sorted(p for p in assets_dir.rglob('*') if p.is_file()):
        rel = f.relative_to(assets_dir).as_posix()
        dest = f'reports/{key}/{ASSET_LINK}/{rel}'
        if not store.is_published(dest):
            store.publish(store.put(f, name=f.name), dest)
    return f'https://huggingface.co/buckets/{store.bucket}/resolve/published/reports/{key}/'


# ---------------------------------------------------------------------------
# Author-link resolution
#
# A report's only *relative* URLs should be its store assets; an author-written link
# (``[src](./experiment.py)``) is repointed by the asset ``<base>`` and would 404. The
# resolver turns each such link into an absolute target — the rendered page for things
# the build renders, the GitHub source otherwise — so it survives the base. In localize
# mode (no base) rendered links stay relative so offline navigation still works.
# ---------------------------------------------------------------------------

_ANCHORED = re.compile(r'(?:[a-z][a-z0-9+.\-]*:|//|/|#)', re.IGNORECASE)


def _repo_slug() -> str | None:
    """``owner/repo`` from ``$MINI_REPO`` or the git ``origin`` remote, or ``None``."""
    url = os.environ.get('MINI_REPO')
    if not url:
        try:
            url = subprocess.run(
                ['git', '-C', str(WORKSPACE_ROOT), 'remote', 'get-url', 'origin'],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except OSError, subprocess.CalledProcessError:
            return None
    m = re.search(r'[:/]([^/]+/[^/]+?)(?:\.git)?$', url)
    return m.group(1) if m else None


@dataclass(frozen=True)
class LinkResolver:
    """Maps an author-written relative link to its published target.

    ``render_map`` is docs-relative *source* path → site-relative *output* path for
    every page the build emits; ``source_files`` is every file under ``docs/`` (the
    GitHub-source fallback). ``site_base``/``source_base`` are the absolute roots used
    when a link must be made absolute (externalize mode).
    """

    render_map: dict[str, str]
    source_files: frozenset[str]
    site_base: str | None
    source_base: str | None
    repo_root: Path | None = None  # used to confirm a link escaping docs/ exists in the repo

    @classmethod
    def discover(cls) -> 'LinkResolver':
        render_map: dict[str, str] = {}
        for md in DOCS_DIR.rglob('*.md'):
            if md.name == 'README.md':
                continue
            rel = md.relative_to(DOCS_DIR).as_posix()
            render_map[rel] = PurePosixPath(rel).with_suffix('.html').as_posix()
        for marimo_dir in DOCS_DIR.rglob('__marimo__'):
            if not marimo_dir.is_dir():
                continue
            rel_parent = marimo_dir.parent.relative_to(DOCS_DIR)
            for html in marimo_dir.rglob('*.html'):
                if ASSET_LINK in html.relative_to(marimo_dir).parts:
                    continue
                stem = html.stem
                out = (rel_parent / f'{stem}.html').as_posix()
                # The export came from a sibling notebook; register every suffix an
                # author might have linked (``report.py`` → its rendered ``report.html``).
                for suffix in _RENDERED_SUFFIXES:
                    render_map[(rel_parent / f'{stem}{suffix}').as_posix()] = out

        source_files = frozenset(p.relative_to(DOCS_DIR).as_posix() for p in DOCS_DIR.rglob('*') if p.is_file())

        slug = _repo_slug()
        site_base = os.environ.get('MINI_SITE_URL')
        source_base = os.environ.get('MINI_SOURCE_URL')
        if slug:
            owner, repo = slug.split('/', 1)
            site_base = site_base or f'https://{owner}.github.io/{repo}/'
            source_base = source_base or f'https://github.com/{slug}/blob/main/'
        return cls(render_map, source_files, site_base, source_base, repo_root=WORKSPACE_ROOT)

    def resolve(self, token: str, *, from_dir: str, externalizing: bool) -> str | None:
        """The rewritten target for relative link *token* (authored under ``docs/<from_dir>``).

        ``None`` means "leave it alone" — an external/absolute link, or one whose target
        the build doesn't know how to reach (a dangling or not-yet-rendered path).
        """
        if not token or _ANCHORED.match(token):
            return None
        path_part, _, frag = token.partition('#')
        frag = f'#{frag}' if frag else ''
        norm = os.path.normpath(PurePosixPath(from_dir, path_part).as_posix())
        if norm.startswith('..'):
            # Escaped docs/, but often still inside the repo — a report linking to its
            # source modules (``../src/experiment``, ``../../src/.../README.md``). Point
            # such a link at the GitHub source so it survives the asset <base> (which
            # would otherwise 404 it against the bucket). Bail if there's no source base,
            # it escapes the repo root too, or the target doesn't exist in the repo.
            if self.source_base is None:
                return None
            repo_rel = os.path.normpath(PurePosixPath('docs', norm).as_posix())
            if repo_rel.startswith('..'):
                return None
            if self.repo_root is not None and not (self.repo_root / repo_rel).exists():
                return None
            return f'{self.source_base}{repo_rel}{frag}'

        if norm in self.render_map:
            if externalizing:
                return None if self.site_base is None else f'{self.site_base}{self.render_map[norm]}{frag}'
            # localize: keep it relative (resolves within _site), just swap the suffix
            return f'{PurePosixPath(path_part).with_suffix(".html").as_posix()}{frag}'
        if norm in self.source_files:
            return None if self.source_base is None else f'{self.source_base}docs/{norm}{frag}'
        return None


def prepare_dirs_and_resolver() -> LinkResolver:
    prepare_dirs()
    return LinkResolver.discover()


# ---------------------------------------------------------------------------


def copy_marimo_output(links: LinkResolver):
    """Copy each ``__marimo__`` report's HTML to ``_site/``, resolving its asset bundle.

    Local store → copy ``_assets/`` beside the HTML (relative URLs resolve in ``_site``).
    HF bucket   → upload ``_assets/`` and insert one ``<base>`` so the same relative
    URLs resolve at the bucket; ``_site`` then carries only HTML. Author-written links
    are resolved to absolute targets either way (so the ``<base>`` doesn't break them).
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
        has_assets = assets_dir.is_dir() and any(assets_dir.rglob('*'))

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
            html = _resolve_html_links(html, links, from_dir=rel_parent.as_posix(), externalizing=externalizing)
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


def _resolve_html_links(html: str, links: LinkResolver, *, from_dir: str, externalizing: bool) -> str:
    """Rewrite resolvable author links in *html*; warn on the ones left dangling."""
    mapping: dict[str, str] = {}
    for token in stray_links(html, link=ASSET_LINK):
        target = links.resolve(token, from_dir=from_dir, externalizing=externalizing)
        if target is not None:
            mapping[token] = target
        else:
            print(f'  ! {from_dir or "."}: unresolved relative link {token!r} — a <base> would break it')
    return rewrite_links(html, mapping) if mapping else html


def copy_assets():
    """Copy non-notebook, non-markdown files from docs/ to _site/."""
    print('Copying assets...')
    skip_dirs = {'__marimo__', '__pycache__'}
    skip_suffixes = {'.py', '.md', '.ipynb', '.pyc', '.pyo'}
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


def _rewrite_md_links(text: str, links: LinkResolver, *, from_dir: str) -> str:
    """Resolve relative Markdown link targets (``](./experiment.py)``) before conversion.

    Markdown pages never carry an asset ``<base>``, so they're resolved in *localize*
    mode: a rendered target stays a relative ``.html`` link (clickable offline), a
    source file becomes an absolute GitHub link, and anything else is left untouched.
    """

    def repl(m: re.Match) -> str:
        token = m.group(1)
        target = links.resolve(token, from_dir=from_dir, externalizing=False)
        return f']({target})' if target is not None else m.group(0)

    return re.sub(r'\]\(([^)\s]+)\)', repl, text)


def convert_markdown(links: LinkResolver):
    """Convert all .md files in docs/ (except README.md) to .html in _site/."""
    print('Converting Markdown...')
    skip = {'README.md'}
    for md_file in sorted(DOCS_DIR.rglob('*.md')):
        if md_file.name in skip:
            continue
        rel = md_file.relative_to(DOCS_DIR).with_suffix('.html')
        dest = SITE_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        from_dir = md_file.parent.relative_to(DOCS_DIR).as_posix()
        text = _rewrite_md_links(md_file.read_text('utf-8'), links, from_dir=from_dir)
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
    links = prepare_dirs_and_resolver()
    copy_marimo_output(links)
    copy_assets()
    copy_md_stylesheet()
    convert_markdown(links)
    add_nojekyll()
    print(f'\nSite written to {SITE_DIR.relative_to(WORKSPACE_ROOT)}/')


if __name__ == '__main__':
    main()

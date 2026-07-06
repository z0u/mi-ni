#!/usr/bin/env python
"""Build the static site from the project's report notebooks.

The HTML lives nowhere in Git: each report is exported (``./go publish``) to a
self-contained bundle — ``index.html`` + named-keyed ``_assets/`` — and mirrored to
the bucket under ``exports/<key>/``. This build is the deterministic, read-only half:
it pulls each synced bundle, resolves author links against the *repo*, inserts one
``<base>`` pointing at the bucket, and writes ``_site/<key>/index.html``. With no
bucket it *localizes* instead — reads the bundles from ``.mini/exports/`` (produced by
``./go export``) and copies their assets beside the HTML so the site works offline.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import markdown as md_lib

from mini.reports import (
    export_dir,
    export_key,
    insert_base,
    report_notebooks,
    rewrite_links,
    set_banner,
    set_theme,
    stray_links,
)

WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
SITE_DIR = WORKSPACE_ROOT / "_site"
DOCS_DIR = WORKSPACE_ROOT / "docs"

# The relative dir, beside each report's index.html, holding its externalized assets
# (figures, data blobs) written by mini.reports.Publisher.
ASSET_LINK = "_assets"

# Source suffixes that the build renders into a report page (so an author link to one
# resolves to the rendered result, not the dead source file).
_RENDERED_SUFFIXES = (".py", ".ipynb", ".md")


def prepare_dirs():
    print("Preparing site directory...")
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()


def _resolve_store():
    """The project store — an HF bucket if configured + authed, else local.

    Drives the two modes: a bucket means *externalize* (pull synced exports + ``<base>``
    at the bucket); a local store means *localize* (read ``.mini/exports`` and copy
    assets beside the HTML in ``_site``).
    """
    from mini.store import store_for

    return store_for(WORKSPACE_ROOT / ".mini" / "store")


# ---------------------------------------------------------------------------
# Author-link resolution
#
# A report's only *relative* URLs should be its store assets; an author-written link
# (``[src](./experiment.py)``) is repointed by the asset ``<base>`` and would 404. The
# resolver turns each such link into an absolute target — the rendered page for things
# the build renders, the GitHub source otherwise — so it survives the base. In localize
# mode (no base) rendered links stay relative so offline navigation still works.
# ---------------------------------------------------------------------------

_ANCHORED = re.compile(r"(?:[a-z][a-z0-9+.\-]*:|//|/|#)", re.IGNORECASE)


def _strip_index(url: str) -> str:
    """Drop a trailing ``index.html`` so a report reads ``<key>/`` not ``<key>/index.html``.

    GitHub Pages serves the directory form, and it's the nicer canonical/shareable URL.
    Operates before any ``#fragment`` and leaves non-index pages (``foo.html``) untouched.
    Used only when publishing — offline (``file://``) navigation keeps the explicit file.
    """
    return re.sub(r"(^|/)index\.html(?=$|#)", r"\1", url)


def _repo_slug() -> str | None:
    """``owner/repo`` from ``$MINI_REPO`` or the git ``origin`` remote, or ``None``."""
    url = os.environ.get("MINI_REPO")
    if not url:
        try:
            url = subprocess.run(
                ["git", "-C", str(WORKSPACE_ROOT), "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except OSError, subprocess.CalledProcessError:
            return None
    m = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else None


@dataclass(frozen=True)
class LinkResolver:
    """Maps an author-written relative link to its published target.

    ``render_map`` is docs-relative *source* path → site-relative *output* path for
    every page the build emits (reports render to ``<key>/index.html``, markdown to
    ``<name>.html``); ``source_files`` is every file under ``docs/`` (the GitHub-source
    fallback). ``site_base``/``source_base`` are the absolute roots used when a link must
    be made absolute (externalize mode).
    """

    render_map: dict[str, str]
    source_files: frozenset[str]
    site_base: str | None
    source_base: str | None
    repo_root: Path | None = None  # used to confirm a link escaping docs/ exists in the repo

    @classmethod
    def discover(cls) -> "LinkResolver":
        render_map: dict[str, str] = {}
        for md in DOCS_DIR.rglob("*.md"):
            if md.name == "README.md":
                continue
            rel = md.relative_to(DOCS_DIR).as_posix()
            render_map[rel] = PurePosixPath(rel).with_suffix(".html").as_posix()
        for nb in report_notebooks(DOCS_DIR):
            out = f"{export_key(nb)}/index.html"
            stem_rel = nb.relative_to(DOCS_DIR)
            # The report came from this notebook; register every suffix an author might
            # have linked (``report.py`` → its rendered ``<key>/index.html``).
            for suffix in _RENDERED_SUFFIXES:
                render_map[stem_rel.with_suffix(suffix).as_posix()] = out

        source_files = frozenset(p.relative_to(DOCS_DIR).as_posix() for p in DOCS_DIR.rglob("*") if p.is_file())

        slug = _repo_slug()
        site_base = os.environ.get("MINI_SITE_URL")
        source_base = os.environ.get("MINI_SOURCE_URL")
        if slug:
            owner, repo = slug.split("/", 1)
            site_base = site_base or f"https://{owner}.github.io/{repo}/"
            source_base = source_base or f"https://github.com/{slug}/blob/main/"
        return cls(render_map, source_files, site_base, source_base, repo_root=WORKSPACE_ROOT)

    def resolve(self, token: str, *, from_dir: str, out_dir: str, externalizing: bool) -> str | None:
        """The rewritten target for relative link *token* authored under ``docs/<from_dir>``.

        The token is interpreted against ``from_dir`` (where it was written); a localized
        link is made relative to ``out_dir`` (where the emitting page *renders*, which for
        a report differs from its source dir). ``None`` means "leave it alone" — an
        external/absolute link, or one whose target the build doesn't know how to reach.
        """
        if not token or _ANCHORED.match(token):
            return None
        path_part, _, frag = token.partition("#")
        frag = f"#{frag}" if frag else ""
        norm = os.path.normpath(PurePosixPath(from_dir, path_part).as_posix())
        if norm.startswith(".."):
            # Escaped docs/, but often still inside the repo — a report linking to its
            # source modules (``../src/experiment``, ``../../src/.../README.md``). Point
            # such a link at the GitHub source so it survives the asset <base> (which
            # would otherwise 404 it against the bucket). Bail if there's no source base,
            # it escapes the repo root too, or the target doesn't exist in the repo.
            if self.source_base is None:
                return None
            repo_rel = os.path.normpath(PurePosixPath("docs", norm).as_posix())
            if repo_rel.startswith(".."):
                return None
            if self.repo_root is not None and not (self.repo_root / repo_rel).exists():
                return None
            return f"{self.source_base}{repo_rel}{frag}"

        if norm in self.render_map:
            out = self.render_map[norm]
            if externalizing:
                return None if self.site_base is None else f"{self.site_base}{_strip_index(out)}{frag}"
            # localize: keep it relative, resolved from where this page renders (out_dir)
            rel = os.path.relpath(out, out_dir or ".")
            return f"{PurePosixPath(rel).as_posix()}{frag}"
        if norm in self.source_files:
            return None if self.source_base is None else f"{self.source_base}docs/{norm}{frag}"
        return None


def prepare_dirs_and_resolver() -> LinkResolver:
    prepare_dirs()
    return LinkResolver.discover()


# ---------------------------------------------------------------------------


def build_reports(links: LinkResolver, store, externalizing: bool):
    """Assemble each report bundle into ``_site/<key>/index.html``.

    Externalize: pull the synced bundle from the bucket, insert one ``<base>`` at
    ``exports/<key>/`` so its relative ``_assets/`` resolve there, and write only the
    HTML into ``_site`` (the bytes stay on the bucket CDN). Localize: read the bundle
    from ``.mini/exports`` and copy its ``_assets/`` beside the HTML so it works offline.
    Author links are resolved to absolute/relative targets either way.
    """
    print("Building reports...")
    for nb in report_notebooks(DOCS_DIR):
        key = export_key(nb)
        from_dir = nb.parent.relative_to(DOCS_DIR).as_posix()  # where author links resolve
        from_dir = "" if from_dir == "." else from_dir
        nb_rel = nb.relative_to(WORKSPACE_ROOT).as_posix()

        with tempfile.TemporaryDirectory() as tmp:
            if externalizing:
                bundle = Path(tmp)
                if not store.fetch_export(key, bundle):
                    print(f"  ! {key}: no synced export on the bucket — run `./go publish` (skipping)")
                    continue
                base_href = store.export_base(key)
            else:
                bundle = export_dir(nb)
                if not (bundle / "index.html").exists():
                    print(f"  ! {key}: not exported locally — run `./go export {nb_rel}` (skipping)")
                    continue
                base_href = None

            html = (bundle / "index.html").read_text("utf-8")
            html = _resolve_html_links(html, links, from_dir=from_dir, out_dir=key, externalizing=externalizing)
            html = set_theme(html)  # follow the visitor's device, not the exporter's setting
            index_url, source_url = _nav_urls(links, key=key, nb_rel=nb_rel, externalizing=externalizing)
            html = set_banner(html, index_url=index_url, source_url=source_url)
            if base_href:
                html = insert_base(html, base_href)
            dest = SITE_DIR / key / "index.html"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(html, "utf-8")

            if not externalizing and (bundle / ASSET_LINK).is_dir():
                shutil.copytree(bundle / ASSET_LINK, dest.parent / ASSET_LINK, dirs_exist_ok=True)
            print(f"  {key} -> _site/{key}/index.html{' [+base]' if base_href else ''}")


def _nav_urls(links: LinkResolver, *, key: str, nb_rel: str, externalizing: bool) -> tuple[str | None, str | None]:
    """The report banner's (index, source) links — same absolute/relative policy as author links.

    The source is the notebook on GitHub (``source_base`` + its repo path). The index is
    the site root: absolute (``site_base``) when externalizing — the asset ``<base>`` would
    otherwise repoint a relative link at the bucket — and relative back up from
    ``_site/<key>/index.html`` when localizing, so offline navigation works. Either is
    ``None`` if its base is unavailable.
    """
    source_url = f"{links.source_base}{nb_rel}" if links.source_base else None
    if externalizing:
        index_url = links.site_base  # the site root serves index.html
    else:
        index_url = "../" * (key.count("/") + 1) + "index.html"
    return index_url, source_url


def _resolve_html_links(html: str, links: LinkResolver, *, from_dir: str, out_dir: str, externalizing: bool) -> str:
    """Rewrite resolvable author links in *html*; warn on the ones left dangling."""
    mapping: dict[str, str] = {}
    for token in stray_links(html, link=ASSET_LINK):
        target = links.resolve(token, from_dir=from_dir, out_dir=out_dir, externalizing=externalizing)
        if target is not None:
            mapping[token] = target
        else:
            print(f"  ! {from_dir or '.'}: unresolved relative link {token!r} — a <base> would break it")
    return rewrite_links(html, mapping) if mapping else html


def copy_assets():
    """Copy non-notebook, non-markdown files from docs/ to _site/."""
    print("Copying assets...")
    skip_dirs = {"__marimo__", "__pycache__"}
    skip_suffixes = {".py", ".md", ".ipynb", ".pyc", ".pyo"}
    for item in sorted(DOCS_DIR.rglob("*")):
        if not item.is_file():
            continue
        parts = item.relative_to(DOCS_DIR).parts
        if any(p in skip_dirs or p.startswith(".") for p in parts):
            continue
        if item.suffix in skip_suffixes:
            continue
        rel = item.relative_to(DOCS_DIR)
        dest = SITE_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {item.relative_to(WORKSPACE_ROOT)} -> {dest.relative_to(WORKSPACE_ROOT)}")
        shutil.copy2(item, dest)


def site_root(dest: Path) -> str:
    """Return the relative path prefix from dest back to the site root."""
    depth = len(dest.relative_to(SITE_DIR).parts) - 1
    return "../" * depth


def copy_md_stylesheet():
    """Copy the Markdown page stylesheet to _site/."""
    print("Copying Markdown stylesheet...")
    css_src = WORKSPACE_ROOT / "scripts" / "md.css"
    css_dest = SITE_DIR / "md.css"
    shutil.copy2(css_src, css_dest)
    print(f"  {css_src.relative_to(WORKSPACE_ROOT)} -> {css_dest.relative_to(WORKSPACE_ROOT)}")


def _rewrite_md_links(text: str, links: LinkResolver, *, from_dir: str, pretty: bool) -> str:
    """Resolve relative Markdown link targets (``](./experiment.py)``) before conversion.

    Markdown pages never carry an asset ``<base>``, so they're resolved in *localize*
    mode: a rendered target stays a relative link (clickable offline), a source file
    becomes an absolute GitHub link, and anything else is left untouched. When publishing
    (``pretty``), a report link drops its ``index.html`` so it reads ``<key>/``; offline
    builds keep the explicit file so ``file://`` navigation still works.
    """

    def repl(m: re.Match) -> str:
        token = m.group(1)
        target = links.resolve(token, from_dir=from_dir, out_dir=from_dir, externalizing=False)
        if target is None:
            return m.group(0)
        return f"]({_strip_index(target) if pretty else target})"

    return re.sub(r"\]\(([^)\s]+)\)", repl, text)


def convert_markdown(links: LinkResolver, externalizing: bool):
    """Convert all .md files in docs/ (except README.md) to .html in _site/."""
    print("Converting Markdown...")
    skip = {"README.md"}
    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        if md_file.name in skip:
            continue
        rel = md_file.relative_to(DOCS_DIR).with_suffix(".html")
        dest = SITE_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        from_dir = md_file.parent.relative_to(DOCS_DIR).as_posix()
        from_dir = "" if from_dir == "." else from_dir
        text = _rewrite_md_links(md_file.read_text("utf-8"), links, from_dir=from_dir, pretty=externalizing)
        body = md_lib.markdown(text, extensions=["extra"])
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else md_file.stem
        root = site_root(dest)
        html = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{title}</title>\n"
            f'<link rel="stylesheet" href="{root}md.css">\n'
            "</head>\n"
            "<body>\n" + body + "\n</body>\n</html>\n"
        )
        dest.write_text(html, "utf-8")
        print(f"  {md_file.relative_to(WORKSPACE_ROOT)} -> {dest.relative_to(WORKSPACE_ROOT)}")


def add_nojekyll():
    (SITE_DIR / ".nojekyll").touch()


def main():
    from mini.hf_store import HFStore

    links = prepare_dirs_and_resolver()
    store = _resolve_store()
    externalizing = isinstance(store, HFStore)
    if isinstance(store, HFStore):  # the publish tier: its own repo if split off, else the bucket
        print(f"  asset mode: externalize ← {store.publish_repo or store.bucket}")
    else:
        print("  asset mode: localize (no bucket)")
    build_reports(links, store, externalizing)
    copy_assets()
    copy_md_stylesheet()
    convert_markdown(links, externalizing)
    add_nojekyll()
    print(f"\nSite written to {SITE_DIR.relative_to(WORKSPACE_ROOT)}/")


if __name__ == "__main__":
    main()

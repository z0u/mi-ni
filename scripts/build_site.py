#!/usr/bin/env python
"""Build the static site from the project's report notebooks.

The HTML lives nowhere in Git: each report is exported (``./go publish``) to a self-contained bundle — ``index.html`` + named-keyed ``_assets/`` — and mirrored to the bucket under ``exports/<key>/``. The assembly mode is an explicit choice, never inferred from credentials:

``--externalize`` (CI, ``./go site``) The deterministic, read-only half of publishing: read each *synced* bundle's HTML, resolve author links against the repo, insert one ``<base>`` pointing at the bucket, and write only ``_site/<key>/index.html`` (asset bytes stay on the CDN). Requires a configured store; fails loudly without one.

``--localize`` (local preview, ``./go preview``) Read the bundles from ``.mini/exports/`` and copy their ``_assets/`` beside the HTML, so the site works offline. Never touches the network.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import markdown as md_lib

from mini.reports import (
    PUBLISH_LOCK,
    ReportFigure,
    export_dir,
    export_key,
    github_slug,
    insert_base,
    load_pins,
    report_figures,
    report_notebooks,
    rewrite_links,
    set_banner,
    set_report_styles,
    set_responsive,
    set_theme,
    stray_links,
)

WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
SITE_DIR = WORKSPACE_ROOT / "_site"
DOCS_DIR = WORKSPACE_ROOT / "docs"

# The shared report stylesheet, re-inlined into every report at build time (see
# mini.reports.set_report_styles). Read from source each build, so editing it restyles
# every published report with no notebook re-export.
REPORT_CSS = DOCS_DIR / "report.css"

# The relative dir, beside each report's index.html, holding its externalized assets
# (figures, data blobs) written by mini.reports.Publisher.
ASSET_LINK = "_assets"

# Mermaid for Markdown pages, pinned to the version Marimo's frontend depends on, so a
# diagram in a .md renders like one `mo.mermaid` draws in a report. Re-check on a marimo
# bump, alongside the font pins in scripts/md.css:
#   curl -s https://cdn.jsdelivr.net/npm/@marimo-team/frontend@<version>/package.json
MERMAID_VERSION = "11.12.3"
MERMAID_URL = f"https://cdn.jsdelivr.net/npm/mermaid@{MERMAID_VERSION}/dist/mermaid.esm.min.mjs"

# Loaded only by a page that holds a diagram, since the bundle is a few MB. Mermaid picks
# up the reader's colour scheme the way md.css does; a scheme changed after load lands on
# the next reload.
MERMAID_SCRIPT = f"""<script type="module">
import mermaid from "{MERMAID_URL}";
const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
mermaid.initialize({{ startOnLoad: false, theme: dark ? "dark" : "default" }});
await mermaid.run();
</script>
"""

# Source suffixes that the build renders into a report page (so an author link to one
# resolves to the rendered result, not the dead source file).
_RENDERED_SUFFIXES = (".py", ".ipynb", ".md")


def prepare_dirs():
    print("Preparing site directory...")
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()


def _resolve_publish_store():
    """The HF publish tier for ``--externalize``, or a loud exit if unreachable.

    Mode is the caller's explicit choice; this only checks the chosen mode is *possible* — it never silently downgrades to localize.
    """
    from mini.hf_store import HFStore
    from mini.store import store_for

    store = store_for(WORKSPACE_ROOT / ".mini" / "store")
    if not isinstance(store, HFStore):
        sys.exit(
            "--externalize needs the HF publish tier (a read token suffices): "
            "set [tool.mini] store-bucket/publish-repo and run `./go auth`.\n"
            "For an offline build from local bundles, use `./go preview` (--localize)."
        )
    return store


# ---------------------------------------------------------------------------
# Author-link resolution
#
# A report's only *relative* URLs should be its store assets; an author-written link
# (``[src](./experiment.py)``) is repointed by the asset ``<base>`` and would 404. The
# resolver turns each such link into an absolute target — the rendered page for things
# the build renders, the GitHub source otherwise — so it survives the base. In localize
# mode (no base) rendered links stay relative so offline navigation still works.
#
# A root-absolute target (``/eng/gc.md``) is the house style for a cross-tree link
# (see https://github.com/z0u/sca2/blob/main/todo/eng/markdown-link-check.md): GitHub and VS Code both read it against
# the repo root, so the resolver does too, rebasing it onto ``docs/`` to take the same
# paths below as a relative one. ``_ANCHORED`` therefore matches only what is already
# absolute or in-page, the same set ``mini.reports`` leaves alone.
# ---------------------------------------------------------------------------

_ANCHORED = re.compile(r"(?:[a-z][a-z0-9+.\-]*:|//|#)", re.IGNORECASE)


def _strip_index(url: str) -> str:
    """Drop a trailing ``index.html`` so a report reads ``<key>/`` not ``<key>/index.html``.

    GitHub Pages serves the directory form, and it's the nicer canonical/shareable URL. Operates before any ``#fragment`` and leaves non-index pages (``foo.html``) untouched. Used only when publishing — offline (``file://``) navigation keeps the explicit file.
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

    ``render_map`` is docs-relative *source* path → site-relative *output* path for every page the build emits (reports render to ``<key>/index.html``, markdown to ``<name>.html``); ``site_assets`` is what :func:`site_asset_files` copies verbatim into ``_site/`` at the same relative path; ``source_files`` is every file under ``docs/`` (the GitHub-source fallback). ``site_base``/``source_base`` are the absolute roots used when a link must be made absolute (externalize mode).
    """

    render_map: dict[str, str]
    source_files: frozenset[str]
    site_base: str | None
    source_base: str | None
    site_assets: frozenset[str] = frozenset()
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
            # have linked (``report.py`` → its rendered ``<key>/index.html``), plus the
            # bare directory (``../ex-2.1.1/``) — the canonical published URL one report
            # naturally uses to link another.
            for suffix in _RENDERED_SUFFIXES:
                render_map[stem_rel.with_suffix(suffix).as_posix()] = out
            render_map[export_key(nb)] = out

        source_files = frozenset(p.relative_to(DOCS_DIR).as_posix() for p in DOCS_DIR.rglob("*") if p.is_file())
        site_assets = frozenset(p.relative_to(DOCS_DIR).as_posix() for p in site_asset_files())

        slug = _repo_slug()
        site_base = os.environ.get("MINI_SITE_URL")
        source_base = os.environ.get("MINI_SOURCE_URL")
        if slug:
            owner, repo = slug.split("/", 1)
            site_base = site_base or f"https://{owner}.github.io/{repo}/"
            source_base = source_base or f"https://github.com/{slug}/blob/main/"
        return cls(render_map, source_files, site_base, source_base, site_assets, repo_root=WORKSPACE_ROOT)

    def _in_site(self, out: str, *, out_dir: str, externalizing: bool, frag: str) -> str | None:
        """How a page rendering into ``out_dir`` should link *out*, a site-relative path.

        Externalizing, the page carries an asset ``<base>`` that would repoint a relative URL at the bucket, so it has to be spelled out from ``site_base`` (and a report reads ``<key>/``, not ``<key>/index.html``). Localizing, it stays relative — resolved from where the page *renders*, which for a report differs from its source dir — so offline navigation works.
        """
        if externalizing:
            return None if self.site_base is None else f"{self.site_base}{_strip_index(out)}{frag}"
        rel = os.path.relpath(out, out_dir or ".")
        return f"{PurePosixPath(rel).as_posix()}{frag}"

    def resolve(self, token: str, *, from_dir: str, out_dir: str, externalizing: bool) -> str | None:
        """The rewritten target for author-written link *token* under ``docs/<from_dir>``.

        A relative token is interpreted against ``from_dir`` (where it was written) and a root-absolute one against the repo root; a localized link is made relative to ``out_dir`` (where the emitting page *renders*, which for a report differs from its source dir). ``None`` means "leave it alone" — an external or in-page link, or one whose target the build doesn't know how to reach.
        """
        if not token or _ANCHORED.match(token):
            return None
        path_part, _, frag = token.partition("#")
        frag = f"#{frag}" if frag else ""
        if path_part.startswith("/"):
            # Root-absolute: repo-root-relative, so rebase onto docs/ and let the
            # branches below decide. One under docs/ renders like any other page;
            # one outside it takes the escape branch and points at the source.
            norm = os.path.relpath(os.path.normpath(path_part.lstrip("/")), "docs")
        else:
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
            return self._in_site(self.render_map[norm], out_dir=out_dir, externalizing=externalizing, frag=frag)
        if norm in self.site_assets:
            # A file the build copies verbatim into _site/ — an image, a data blob. The
            # site serves it at the same relative path, so point there. The GitHub
            # fallback below would hand back a ``blob/`` URL, which is an HTML page
            # rather than the bytes: fine to click, but an ``<img>`` renders nothing.
            return self._in_site(norm, out_dir=out_dir, externalizing=externalizing, frag=frag)
        if norm in self.source_files:
            return None if self.source_base is None else f"{self.source_base}docs/{norm}{frag}"
        return None


def prepare_dirs_and_resolver() -> LinkResolver:
    prepare_dirs()
    return LinkResolver.discover()


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Bundle:
    """One report's exported HTML as read from its source, before any assembly.

    ``html`` is ``None`` when there's nothing to assemble (never published, or never exported locally). ``notes`` are log lines the read wants printed — held here rather than printed on the spot, so a concurrent read still logs in notebook order.
    """

    html: str | None
    base_href: str | None = None  # externalize: the CDN dir the report's _assets/ resolve against
    assets: Path | None = None  # localize: the local _assets/ dir to copy beside the HTML
    notes: tuple[str, ...] = ()


def _read_bundle(nb: Path, *, store, pins: dict[str, str], externalizing: bool) -> _Bundle:
    """Read one report's exported ``index.html`` — off the bucket, or from ``.mini/exports/``.

    Externalize reads *only* the HTML. The page's ``_assets/`` links stay relative and the ``<base>`` sends them to the bundle on the CDN, so pulling the whole bundle here would fetch megabytes of figures the build has no use for. It's also the one step that waits on the network, which is why it's separable: the reports are independent, so the caller runs these together instead of serially.

    A report pinned in ``docs/publish.lock`` is read *and* based at that revision, so the page serves exactly what its publish uploaded — a later re-publish (e.g. from a branch whose PR hasn't merged) can't swap the assets under this build. An unpinned report falls back to the mutable branch head, with a warning.
    """
    key = export_key(nb)
    if not externalizing:
        bundle = export_dir(nb)
        if not (bundle / "index.html").exists():
            nb_rel = nb.relative_to(WORKSPACE_ROOT).as_posix()
            return _Bundle(None, notes=(f"  ! {key}: not exported locally — run `./go preview {nb_rel}` (skipping)",))
        assets = bundle / ASSET_LINK
        return _Bundle((bundle / "index.html").read_text("utf-8"), assets=assets if assets.is_dir() else None)

    notes: list[str] = []
    revision = pins.get(key)
    # Only a git-backed publish tier can pin; on the single-bucket default the mutable
    # head is all there is, so the nudge would be misleading.
    if revision is None and store.publish_repo is not None:
        notes.append(f"  ! {key}: not pinned in {PUBLISH_LOCK} — serving the mutable head; `./go publish` to pin")
    html = store.read_export_html(key, revision=revision)
    if html is None:
        notes.append(f"  ! {key}: no synced export on the bucket — run `./go publish` (skipping)")
        return _Bundle(None, notes=tuple(notes))
    return _Bundle(html, base_href=store.export_base(key, revision=revision), notes=tuple(notes))


@dataclass(frozen=True)
class FigureStrip:
    """One built report's figures, for a Markdown page to render as thumbnails.

    ``base_href`` is the CDN dir the bundle's relative asset URLs resolve against — the same (revision-pinned) base the report page itself gets — or ``None`` when localizing, where the assets sit at ``_site/<key>/_assets/`` and a page links them relatively.
    """

    key: str
    base_href: str | None
    figures: tuple[ReportFigure, ...]


def build_reports(links: LinkResolver, store, externalizing: bool) -> dict[str, FigureStrip]:
    """Assemble each report bundle into ``_site/<key>/index.html``.

    Externalize: read the synced HTML from the bucket, insert one ``<base>`` at ``exports/<key>/`` so its relative ``_assets/`` resolve there, and write only the HTML into ``_site`` (the bytes stay on the bucket CDN). Localize: read the bundle from ``.mini/exports`` and copy its ``_assets/`` beside the HTML so it works offline. Author links are resolved to absolute/relative targets either way.

    Returns each built report's :class:`FigureStrip` by key, so :func:`convert_markdown` can expand ``mini:figures`` markers from the HTML this pass already fetched.
    """
    print("Building reports...")
    pins = load_pins(WORKSPACE_ROOT) if externalizing else {}
    report_css = REPORT_CSS.read_text("utf-8") if REPORT_CSS.exists() else ""
    nbs = report_notebooks(DOCS_DIR)
    # Externalized, each read is a round trip to the bucket and the reports don't depend
    # on each other — so read them in one wave and the build waits for the slowest report
    # rather than the sum of all of them. Assembly below is CPU-cheap and stays sequential
    # in notebook order, so the log reads the same however the threads interleaved.
    with ThreadPoolExecutor(max_workers=min(8, max(len(nbs), 1))) as pool:
        bundles = pool.map(lambda nb: _read_bundle(nb, store=store, pins=pins, externalizing=externalizing), nbs)

    strips: dict[str, FigureStrip] = {}
    for nb, bundle in zip(nbs, list(bundles), strict=True):
        key = export_key(nb)
        for note in bundle.notes:
            print(note)
        if bundle.html is None:
            continue
        strips[key] = FigureStrip(key, bundle.base_href, tuple(report_figures(bundle.html, link=ASSET_LINK)))
        from_dir = nb.parent.relative_to(DOCS_DIR).as_posix()  # where author links resolve
        from_dir = "" if from_dir == "." else from_dir
        nb_rel = nb.relative_to(WORKSPACE_ROOT).as_posix()

        html = _resolve_html_links(bundle.html, links, from_dir=from_dir, out_dir=key, externalizing=externalizing)
        html = set_theme(html)  # follow the visitor's device, not the exporter's setting
        html = set_responsive(html)  # fit narrow screens; drop Marimo's watermark
        index_url, source_url = _nav_urls(links, key=key, nb_rel=nb_rel, externalizing=externalizing)
        html = set_banner(html, index_url=index_url, source_url=source_url)
        html = set_report_styles(html, report_css)  # last, so shared report rules win ties
        if bundle.base_href:
            html = insert_base(html, bundle.base_href)
        dest = SITE_DIR / key / "index.html"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(html, "utf-8")

        if bundle.assets is not None:
            shutil.copytree(bundle.assets, dest.parent / ASSET_LINK, dirs_exist_ok=True)
        print(f"  {key} -> _site/{key}/index.html{' [+base]' if bundle.base_href else ''}")
    return strips


def _nav_urls(links: LinkResolver, *, key: str, nb_rel: str, externalizing: bool) -> tuple[str | None, str | None]:
    """The report banner's (index, source) links — same absolute/relative policy as author links.

    The source is the notebook on GitHub (``source_base`` + its repo path). The index is the site root: absolute (``site_base``) when externalizing — the asset ``<base>`` would otherwise repoint a relative link at the bucket — and relative back up from ``_site/<key>/index.html`` when localizing, so offline navigation works. Either is ``None`` if its base is unavailable.
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


_ASSET_SKIP_DIRS = {"__marimo__", "__pycache__"}
_ASSET_SKIP_SUFFIXES = {".py", ".md", ".ipynb", ".pyc", ".pyo"}


def site_asset_files() -> list[Path]:
    """Files under ``docs/`` the build copies verbatim into ``_site/``, at the same relative path.

    One definition, read twice: :func:`copy_assets` copies them, and :class:`LinkResolver` needs the same set to know that a link to one is served by the site. Were the two to drift, an author link would resolve somewhere the copy never put the file.
    """
    out = []
    for item in sorted(DOCS_DIR.rglob("*")):
        if not item.is_file() or item == WORKSPACE_ROOT / PUBLISH_LOCK:  # the pin manifest is build input, not content
            continue
        parts = item.relative_to(DOCS_DIR).parts
        if any(p in _ASSET_SKIP_DIRS or p.startswith(".") for p in parts):
            continue
        if item.suffix in _ASSET_SKIP_SUFFIXES:
            continue
        out.append(item)
    return out


def copy_assets():
    """Copy non-notebook, non-markdown files from docs/ to _site/."""
    print("Copying assets...")
    for item in site_asset_files():
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

    Markdown pages never carry an asset ``<base>``, so they're resolved in *localize* mode: a rendered target stays a relative link (clickable offline), a source file becomes an absolute GitHub link, and anything else is left untouched. When publishing (``pretty``), a report link drops its ``index.html`` so it reads ``<key>/``; offline builds keep the explicit file so ``file://`` navigation still works.
    """

    def repl(m: re.Match) -> str:
        token = m.group(1)
        target = links.resolve(token, from_dir=from_dir, out_dir=from_dir, externalizing=False)
        if target is None:
            return m.group(0)
        return f"]({_strip_index(target) if pretty else target})"

    return re.sub(r"\]\(([^)\s]+)\)", repl, text)


def render_markdown(text: str) -> str:
    """A Markdown page's HTML body, with a GitHub-compatible ``id`` on every heading.

    ``toc`` is what puts the ids there — without it a heading renders bare, so a ``#fragment`` into a page works on GitHub and scrolls nowhere here, which no link check can see from the source alone. Its own slugify collapses a run of separators, so it has to be handed :func:`github_slug` instead or the site would speak a third dialect: ``check_md_links`` validates a fragment against GitHub's slugs, and a link that resolves there has to resolve here.
    """
    return md_lib.markdown(
        text,
        extensions=["extra", "md_in_html", "toc"],
        extension_configs={"toc": {"slugify": lambda value, separator: github_slug(value)}},
    )


# A figure-strip marker in a Markdown page: `<!-- mini:figures ./pipeline/report.py -->`
# on its own line, naming a report the way an ordinary link would. Comments pass through
# both GitHub's renderer (invisible there — the source view stays clean) and
# Python-Markdown (so the marker survives into the rendered body, where the build swaps
# it for the strip), and `check_md_links` strips comments before matching, so the path
# inside costs nothing there either.
_FIGURES_MARKER = re.compile(r"<!--\s*mini:figures\s+(\S+)\s*-->")


def _marker_key(token: str, links: LinkResolver, *, from_dir: str) -> str | None:
    """The export key a ``mini:figures`` marker names, or ``None`` for an unknown target.

    Accepts the same spellings a link to the report would use — ``./pipeline/report.py`` or the bare directory — resolved against the page's own dir via the resolver's ``render_map``, so the marker can't drift from how the rest of the page addresses reports.
    """
    norm = os.path.normpath(PurePosixPath(from_dir, token).as_posix())
    out = links.render_map.get(norm)
    suffix = "/index.html"
    return out[: -len(suffix)] if out is not None and out.endswith(suffix) else None


def _figure_strip_html(strip: FigureStrip, *, from_dir: str, externalizing: bool) -> str:
    """A report's thumbnail strip: each figure a lazy full-size image, themed via ``<picture>``.

    Externalizing, URLs use the strip's revision-pinned CDN base — the same assets the report page serves, so the index can never show figures its report doesn't. Localizing they're relative into the copied ``_site/<key>/_assets/``. Each thumbnail reuses the figure's own alt text and the ``width``/``height`` the export stamped (so the row lays out before the PNGs arrive). The CSS (``scripts/md.css``) sizes them down, so the browser fetches one theme's PNG per figure and only as it scrolls into view. Deliberately no link to the full-size PNG: opening one paints its transparent background on the browser's white canvas, wrong in dark mode, while copying the image in place picks up the scheme-matched variant the ``<picture>`` shows.
    """
    import html

    base = (
        strip.base_href
        if externalizing
        else f"{PurePosixPath(os.path.relpath(strip.key, from_dir or '.')).as_posix()}/"
    )
    parts = []
    for fig in strip.figures:
        alt, title = html.escape(fig.alt), html.escape(fig.stem)
        size = f' width="{fig.width}" height="{fig.height}"' if fig.width and fig.height else ""
        img = f'<img src="{base}{fig.light}" alt="{alt}" title="{title}"{size} loading="lazy">'
        if fig.dark:
            img = f'<picture><source media="(prefers-color-scheme: dark)" srcset="{base}{fig.dark}">{img}</picture>'
        parts.append(img)
    return f'<div class="fig-strip">{"".join(parts)}</div>'


def expand_figure_strips(
    body: str, strips: dict[str, FigureStrip], links: LinkResolver, *, from_dir: str, externalizing: bool
) -> str:
    """Swap each ``mini:figures`` marker in a rendered page for its report's thumbnail strip.

    Operates on the rendered HTML rather than the Markdown, so the strip's markup never has to survive Python-Markdown's block parsing (it would read a raw ``<div>`` indented inside a list item as code). A marker whose report wasn't built (unpublished, or skipped) renders as nothing, with a build note.
    """

    def repl(m: re.Match) -> str:
        token = m.group(1)
        strip = strips.get(_marker_key(token, links, from_dir=from_dir) or "")
        if strip is None:
            print(f"  ! {from_dir or '.'}: mini:figures {token!r} names no built report — dropping the strip")
            return ""
        return _figure_strip_html(strip, from_dir=from_dir, externalizing=externalizing)

    return _FIGURES_MARKER.sub(repl, body)


_MERMAID_FENCE = re.compile(r'<pre><code class="language-mermaid">(.*?)</code></pre>', re.DOTALL)


def promote_mermaid(html: str) -> tuple[str, bool]:
    """Rewrite a ```mermaid fence into the ``<pre class="mermaid">`` the library renders into, and say whether the page has one.

    Python-Markdown renders any fence as a nested ``<pre><code>``, which mermaid walks straight past; the flag then keeps its script off every page that holds no diagram. The escaping the fence applied (``&quot;`` for a node label, ``&amp;`` for a fan-out edge) is left in place — the parser reads the element's text, which the browser has already decoded.
    """
    html, count = _MERMAID_FENCE.subn(r'<pre class="mermaid">\1</pre>', html)
    return html, bool(count)


def convert_markdown(links: LinkResolver, externalizing: bool, strips: dict[str, FigureStrip] | None = None):
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
        body, has_mermaid = promote_mermaid(render_markdown(text))
        body = expand_figure_strips(body, strips or {}, links, from_dir=from_dir, externalizing=externalizing)
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
            f'<link rel="stylesheet" href="{root}md.css">\n' + (MERMAID_SCRIPT if has_mermaid else "") + "</head>\n"
            "<body>\n" + body + "\n</body>\n</html>\n"
        )
        dest.write_text(html, "utf-8")
        print(f"  {md_file.relative_to(WORKSPACE_ROOT)} -> {dest.relative_to(WORKSPACE_ROOT)}")


def add_nojekyll():
    (SITE_DIR / ".nojekyll").touch()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--externalize",
        action="store_true",
        help="assemble from published bundles; assets stay on the CDN behind a <base> (CI)",
    )
    mode.add_argument(
        "--localize", action="store_true", help="assemble from .mini/exports/ with assets copied in; works offline"
    )
    args = ap.parse_args()

    # Resolve the store *before* wiping _site, so a missing token can't destroy a build.
    if args.externalize:
        store = _resolve_publish_store()
        print(f"  asset mode: externalize ← {store.publish_repo or store.bucket}")
    else:
        store = None
        print("  asset mode: localize (.mini/exports/)")
    links = prepare_dirs_and_resolver()
    strips = build_reports(links, store, args.externalize)
    copy_assets()
    copy_md_stylesheet()
    convert_markdown(links, args.externalize, strips)
    add_nojekyll()
    print(f"\nSite written to {SITE_DIR.relative_to(WORKSPACE_ROOT)}/")


if __name__ == "__main__":
    main()

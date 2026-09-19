"""
Report bundles: produce a report's assets as relative URLs, then repoint them.

A report is a **bundle** — one HTML document (woven by ``mini.lit``) plus its heavy assets (figures, data blobs). The two halves of the bundle protocol both live here:

**Produce.** A :class:`Publisher` writes each asset out as a file beside the exported HTML and hands back a *relative* URL like ``_assets/<name>.png``. The path is the asset's readable name (so a browser saving it suggests a sensible filename — the URL's last segment, since the bucket sets no ``Content-Disposition``), and the name *is* the key, so a re-render overwrites in place and the URL stays stable. ``themed`` figures externalize through a publisher when one is set; :meth:`Publisher.asset_url` is the general verb for any blob.

**Publish.** That same HTML is consumed two ways:

- **opened locally**, the relative URL resolves to the co-located ``_assets/`` files;
- **served from Pages**, we want it to resolve to the assets we uploaded to the HF bucket instead.

The bridge is a single ``<base href>`` in the ``<head>`` (:func:`insert_base`): it sets the document base that *every* relative URL resolves against, so one inserted tag repoints the whole report's assets at the bucket — no per-URL rewriting, and it works for a relative ``fetch()`` too.

The catch is that ``<base>`` is document-global, so an author-written relative *link* (a markdown ``[src](./experiment.py)``) would be repointed too — and 404 against the bucket. :func:`stray_links` finds those at build time; :func:`rewrite_links` turns them into absolute targets (their rendered page, or their source) so they survive the base. The convention is *the only relative URLs left in a report are store assets*.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import types
import unicodedata
from dataclasses import dataclass, field
from html import escape as html_escape, unescape as html_unescape
from pathlib import Path, PurePosixPath
from typing import Any, cast

from mini.store import active_profile

__all__ = [
    "Publisher",
    "export_key",
    "export_dir",
    "input_dir",
    "inputs_touched_at",
    "is_stale",
    "PUBLISH_LOCK",
    "load_pins",
    "save_pins",
    "is_report",
    "reports",
    "is_manually_published",
    "SOURCE_ONLY_MARKER",
    "MANUAL_PUBLISH_MARKER",
    "PROVENANCE_ASSET",
    "use_publisher",
    "current_publisher",
    "externalize_html",
    "link_externalized",
    "ASSET_MARKER",
    "relative_urls",
    "stray_links",
    "ReportFigure",
    "report_figures",
    "write_thumbnails",
    "mark_figures",
    "set_lightbox",
    "lightbox_chrome",
    "ZOOM_ATTR",
    "THUMBNAIL_META",
    "THUMBNAIL_DIR",
    "THUMBNAIL_HEIGHT",
    "rewrite_links",
    "github_slug",
    "insert_base",
    "set_report_styles",
    "set_banner",
    "set_provenance",
    "set_alternate",
    "alternates",
    "PDF_LEAF",
    "PDF_TYPE",
    "MD_LEAF",
    "MD_TYPE",
]

# Markers that identify the project root (mirrors mini.runs._ROOT_MARKERS).
_ROOT_MARKERS = ("pyproject.toml", ".git")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Produce: writing a report's assets as files referenced by a relative URL
# ---------------------------------------------------------------------------


def _safe_leaf(name: str) -> str:
    """A filesystem/URL-safe leaf filename from *name* (its readable download name)."""
    leaf = re.sub(r"[^A-Za-z0-9._-]", "-", PurePosixPath(name).name)
    return leaf or "asset"


# The bundle's provenance sidecar: which store refs the report resolved when it was
# rendered, and the producer stamped on each (see ``mini.store.Store.set_ref``). The
# publisher maintains it as the report runs; the exporter reads it back to inject
# the report's provenance footer. It lives in ``_assets/`` so it rides the bundle
# sync — the published site carries its own machine-readable provenance.
PROVENANCE_ASSET = "provenance.json"


@dataclass(frozen=True)
class Publisher:
    """Writes a report's heavy assets out as files beside the exported HTML, referenced by a **relative** URL.

    Each blob is written under ``asset_dir`` (the report's bundle ``_assets/``) at its readable *name*. The name *is* the key, so the URL is stable across re-exports — a re-render overwrites in place rather than piling up a new content-addressed copy each time (which is what kept the bucket accumulating orphans). The name is also what a browser "Save as" suggests (it derives the filename from the URL's last segment, the bucket setting no ``Content-Disposition``). The reference is ``<link>/<name>``; because it's relative, the same HTML resolves to the local files when opened off disk and to the HF bucket when published (a single ``<base href>`` is inserted at build time — see ``scripts/build_site.py``).
    """

    asset_dir: Path
    link: str = "_assets"
    # Whether a second, *different* blob under one name is an error. It is during an
    # export (one pass, so two figures are colliding on a name), but it's the ordinary
    # edit loop under ``mini.lit``'s live server — re-running a figure cell after
    # tweaking the plot writes new bytes under the same name, and should just replace it.
    strict: bool = True
    # Whether the returned URL carries a ``?v=<content hash>``. A stable filename is the
    # point of the naming scheme, but it means an edited figure keeps its URL, so a
    # browser goes on showing the copy it already has (the live server sends no
    # ``Cache-Control``, leaving the freshness heuristic to guess, and it guesses stale).
    # Stamping the hash makes changed bytes a new URL and unchanged bytes the same one —
    # so the cache still does its job between edits. Off for an export: a published
    # bundle already gets a fresh URL per revision (the ``<base href>`` carries the
    # commit sha), and a query string there would only churn the HTML.
    versioned: bool = False
    # name -> sha of what we wrote under it this export, so a second *different*
    # blob under the same name is caught rather than silently clobbering.
    _written: dict[str, str] = field(default_factory=dict, compare=False, repr=False)
    # ref name -> the producer stamped on it (or None) — every store ref the report
    # resolved while rendering, mirrored to the PROVENANCE_ASSET sidecar.
    _refs: dict[str, dict[str, Any] | None] = field(default_factory=dict, compare=False, repr=False)
    # Every leaf written, in order — so a caller can tell which assets one stretch of
    # work produced (``mini.lit.caching`` records them beside a cached value).
    log: list[str] = field(default_factory=list, compare=False, repr=False)

    def note_ref(self, name: str, producer: dict[str, Any] | None) -> None:
        """Record that the report resolved store ref *name*, written by *producer*.

        Called by ``mini.store`` on every ``get_ref`` while this publisher is active, so the bundle's provenance sidecar always reflects the refs the *current* render actually read. Deterministic given the store's refs — re-rendering unchanged data rewrites the same sidecar.
        """
        self._refs[name] = producer
        dest = self.asset_dir / PROVENANCE_ASSET
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{PROVENANCE_ASSET}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"refs": self._refs}, sort_keys=True, indent=1))
        tmp.replace(dest)

    def asset_url(self, data: bytes | Path, *, name: str, serve: bool = True) -> str:
        """Write *data* (bytes or a file) as ``<name>`` and return its URL.

        Pass ``serve=False`` for a blob written only so that tooling can read it off disk (see :func:`externalize_html`); the URL is still returned.

        The asset is keyed by its readable *name* (carry the extension — it sets the served media type), so the URL is stable and a re-render overwrites in place. Under ``strict`` (the default, and what an export uses) two *different* blobs written under the same name is an authoring bug — give each figure a distinct ``name=`` — so it raises rather than clobber. Under ``versioned`` the URL carries a ``?v=`` stamp of the content, so a re-render is visible through a browser cache.
        """
        blob = bytes(data) if isinstance(data, (bytes, bytearray)) else Path(data).read_bytes()
        leaf = _safe_leaf(name)
        if leaf == PROVENANCE_ASSET:
            raise ValueError(f"{PROVENANCE_ASSET!r} is reserved for the bundle's provenance sidecar")
        sha = hashlib.sha256(blob).hexdigest()
        if self.strict and (prev := self._written.get(leaf)) is not None and prev != sha:
            raise ValueError(
                f"two different assets written as {leaf!r} in one report — pass a distinct "
                "name= to disambiguate (the asset name is the stable URL now, with no content hash)"
            )
        self._written[leaf] = sha
        self.log.append(leaf)
        dest = self.asset_dir / leaf
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{leaf}.{os.getpid()}.tmp")  # per process, so concurrent writers never share a temp file
        tmp.write_bytes(blob)
        tmp.replace(dest)  # atomic + overwrite-in-place: a re-render replaces, never piles up
        return f"{self.link}/{leaf}?v={sha[:8]}" if self.versioned else f"{self.link}/{leaf}"


def _project_root(start: Path) -> Path:
    """The project root (nearest ``pyproject.toml`` / ``.git``) walking up from *start*.

    Anchored at the *path*, not the cwd, so it's stable during an export (which may run from anywhere) — ``__file__`` is absolute there.
    """
    start = start.resolve()
    for d in (start, *start.parents):
        if any((d / m).exists() for m in _ROOT_MARKERS):
            return d
    return start.parent


def export_key(report: str | Path) -> str:
    """The docs-relative, suffix-less key naming a report's self-contained bundle.

    ``docs/gpt.py`` → ``gpt``; ``docs/gpt-sweep/report.py`` → ``gpt-sweep``. A report named ``report.py`` takes its *directory* as the key, so the common one-experiment, one-report split publishes at ``gpt-sweep/`` rather than the redundant ``gpt-sweep/report/``. A second report alongside it keeps its own stem (``docs/foo/aside.py`` → ``foo/aside``), so the convention extends to multiple reports per experiment without collision. The key names the report's on-disk export dir *and* its ``exports/<key>/`` prefix on the bucket, and (served as ``index.html``) its URL ``<key>/`` — so each report is one independently syncable bundle.
    """
    p = Path(report).resolve()
    docs = _project_root(p) / "docs"
    try:
        rel = p.relative_to(docs)
    except ValueError:
        rel = Path(p.name)
    key = rel.with_suffix("")
    if key.name == "report" and key.parent != Path("."):
        key = key.parent  # a directory's canonical report drops the redundant /report
    return key.as_posix()


def export_dir(report: str | Path) -> Path:
    """The local (gitignored) dir holding a report's exported ``index.html`` + ``_assets/``.

    ``<root>/.mini/exports/<key>/`` — the unit that mirrors to bucket ``exports/<key>/``. Kept under ``.mini`` (already gitignored) so exported HTML never enters Git.
    """
    p = Path(report).resolve()
    return _project_root(p) / ".mini" / "exports" / export_key(p)


def input_dir(report: str | Path) -> Path | None:
    """The directory whose files are this report's *local* inputs, or ``None`` if it has none.

    The mirror of :func:`export_dir`: that names what a report writes, this names what it reads from the repo. A report that owns a directory (``docs/ex-2.1.8/report.py``) reads the files beside it — the ``experiment.py`` defining its tasks, a ``dopesheet.csv``, whatever else the author put there — so an edit to any of them dates the report's bundle exactly as an edit to the script does. Callers that only watch the ``.py`` see a report re-run against new results and report itself unchanged.

    Scoped to the directory rather than a parsed import graph because the directory *is* the convention here (:func:`export_key` already derives a report's identity from it), and it stays right without anyone maintaining it. It's deliberately the loose end of the two: a shared module under ``src/`` is an input too, and nothing local can see that — the bundle's ``PROVENANCE_ASSET`` sidecar is where that question gets answered, at the cost of store access.

    ``None`` for a report living directly in ``docs/`` (``docs/overview.py``): the docs root is shared site space — ``publish.lock``, ``index.md``, ``report.css`` — not one report's inputs, and reading it as such would date every root-level report on every publish.
    """
    parent = Path(report).resolve().parent
    docs = _project_root(parent) / "docs"
    return parent if parent != docs and docs in parent.parents else None


def inputs_touched_at(report: str | Path) -> float:
    """The mtime of the most recently edited thing *report* is built from.

    The script, plus everything in its input directory (:func:`input_dir`) — an experiment definition, a dopesheet — since editing one of those dates any render of the report exactly as editing the script does. Directories are stamped too, so deleting an input registers (a delete bumps the parent's mtime while touching no surviving file).

    Skips ``__pycache__`` and dotfiles: importing ``experiment.py`` rewrites its bytecode, which would otherwise read as an edit and re-render the report every time something imported it. The *first* such import still registers, since creating ``__pycache__/`` stamps the directory holding it — one spurious re-render per fresh checkout, which is the price of noticing deletes at all.
    """
    script = Path(report).resolve()
    paths = [script]
    if d := input_dir(script):
        junk = (".", "__pycache__")
        paths += [d, *(p for p in d.rglob("*") if not any(s.startswith(junk) for s in p.relative_to(d).parts))]
    return max(p.stat().st_mtime for p in paths if p.exists())


def is_stale(report: str | Path, output: Path) -> bool:
    """Whether *output* is missing or older than anything *report* is built from (:func:`inputs_touched_at`).

    A cheap mtime heuristic for the bundle's ``index.html`` (``./go preview --stale-only``, the default). It misses edits to imported ``src/`` modules and to the stored results a report reads, so callers offer a ``--force`` that skips the check.
    """
    out = Path(output)
    return not out.exists() or out.stat().st_mtime < inputs_touched_at(report)


# The pin manifest: export key → the publish-tier commit sha its bundle was last
# published as. It lives in Git (not the store) because *that placement is the
# mechanism*: publishing from a branch mints an immutable revision on the dataset repo
# but only changes the pin on that branch — production keeps serving main's pins, the
# PR preview serves the branch's, and merging the PR is what promotes. The identity
# (which revision a report serves at) travels with the code; the store holds only
# evidence (the bundles, at every revision ever published).
PUBLISH_LOCK = Path("docs") / "publish.lock"


def publish_lock(profile: str | None | types.EllipsisType = ...) -> Path:
    """The pin manifest's project-relative path for *profile*: ``...`` (default) means the active one.

    :data:`PUBLISH_LOCK` is the production identity record — the file that says which revision the site serves — so only a publish to the production pair may write it. Under a storage profile (:func:`~mini.store.active_profile`) the pins go to a gitignored ``.mini/publish.<profile>.lock`` instead: a dev publish then changes nothing CI can see, and a dev publish of a report someone meant for production leaves its production pin unmoved, which the unpublished-reports check flags.
    """
    if profile is ...:
        profile = active_profile()
    return PUBLISH_LOCK if profile is None else Path(".mini") / f"publish.{profile}.lock"


def load_pins(project_root: str | Path, *, profile: str | None | types.EllipsisType = ...) -> dict[str, str]:
    """The pin manifest — export key → publish-tier revision — or ``{}`` if none yet.

    *profile* picks the manifest (:func:`publish_lock`): the active profile's by default, or ``None`` for production's regardless of the environment — what a check of production state must ask for.
    """
    path = Path(project_root) / publish_lock(profile)
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def save_pins(
    project_root: str | Path, pins: dict[str, str], *, profile: str | None | types.EllipsisType = ...
) -> Path:
    """Write the pin manifest (sorted, one key per line — Git merges stay trivial)."""
    path = Path(project_root) / publish_lock(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(pins.items())), indent=1) + "\n", "utf-8")
    return path


# A docs script carrying this marker is a source-only *example*, not a rendered
# report: the build skips it (never runs its inline compute) and links to it resolve
# to its GitHub source instead of a site page. For scripts that don't fit the
# read-from-store report model — e.g. ``docs/gpt.py`` trains inline on every run, so
# exporting it would re-run the whole experiment. Put it in a comment right under the
# ``# title:`` line; the text is matched literally.
SOURCE_ONLY_MARKER = "mini:source-only"


def is_report(path: str | Path) -> bool:
    """Whether *path* is a report the site renders.

    A report is a literate script (``mini.lit``: a ``.py`` opening with a ``# title:`` header) that is *not* flagged ``# mini:source-only`` (:data:`SOURCE_ONLY_MARKER`): the marker opts a script out of the published set, so the build neither runs nor renders it and links to it fall back to its GitHub source. The scripts are the only source of truth for the report set — a report is on the site iff its ``.py`` is in the repo and its bundle is synced.
    """
    p = Path(path)
    if p.suffix != ".py" or not p.is_file():
        return False
    from mini.lit import is_literate_script  # here rather than at the top: mini.lit builds on this module

    return is_literate_script(p) and SOURCE_ONLY_MARKER not in p.read_text("utf-8", errors="ignore")


def reports(docs: str | Path) -> list[Path]:
    """Every report under *docs* (sorted); see :func:`is_report`."""
    return sorted(p for p in Path(docs).rglob("*.py") if is_report(p))


# A report carrying this marker is republished on a schedule its author controls, so
# nothing reminds you when an edit leaves its bundle behind (the pre-push hook and CI's
# publish check both skip it). It stays a full report otherwise: rendered, pinned, on the
# site. Like :data:`SOURCE_ONLY_MARKER` the text is matched literally, so put it in a
# comment near the top of the script.
MANUAL_PUBLISH_MARKER = "mini:manual-publish"


def is_manually_published(path: str | Path) -> bool:
    """Whether *path* opts out of the "you changed this without republishing" reminder.

    The two markers answer different questions: :data:`SOURCE_ONLY_MARKER` says "not a report at all", :data:`MANUAL_PUBLISH_MARKER` says "a report, but I'll decide when it publishes". See :func:`is_report` for the first.
    """
    return MANUAL_PUBLISH_MARKER in Path(path).read_text("utf-8", errors="ignore")


_default_publisher: Publisher | None = None


def use_publisher(publisher: Publisher | None) -> Publisher | None:
    """Set the report-wide default publisher; ``mini.lit`` does this once per render.

    Every ``@themed`` figure then externalizes through it with no per-figure argument. Pass a :class:`Publisher`, or ``None`` to clear it (figures inline as self-contained ``data:`` URIs). Returns it, e.g. to call :meth:`~Publisher.asset_url` for a data blob.
    """
    global _default_publisher
    _default_publisher = publisher
    return publisher


def current_publisher() -> Publisher | None:
    """The report-wide default publisher set by :func:`use_publisher` (or ``None``)."""
    return _default_publisher


# Stamped on an externalized fragment's root element, carrying that fragment's sidecar
# URL. Nothing in the browser reads it — it is there for :func:`link_externalized`,
# which swaps the whole element for a link to the sidecar in the Markdown rendition
# rather than carry a page of path data. Because nothing fetches it, it can stay the
# bundle-relative URL; ``insert_base`` and :func:`stray_links` both look at
# ``src``/``href`` only, so it rides along untouched.
ASSET_MARKER = "data-mini-asset"

# The opening tag of a fragment's root element: leading whitespace, ``<``, a tag name.
_ROOT_TAG = re.compile(r"\s*<[a-zA-Z][\w.:-]*")


def externalize_html(fragment: str, *, name: str, publish: Publisher | None = None) -> str:
    """Write *fragment* (an HTML/SVG chunk) out as a named bundle asset, and return it stamped for inlining.

    The inline copy is the one readers see — an inlined SVG participates in the page's CSS (theming, fonts), which a referenced file can't. But an inlined fragment is a screenful of path data in the page, so tooling that reads the document as text has to wade through it. The sidecar under ``_assets/`` is the escape hatch: the same fragment as a plain file, like the PNGs ``themed`` writes. *name* keeps its extension if it has one (``.svg`` for a bare SVG element), else ``.html``. With no publisher (*publish* or the report default), this is a no-op pass-through.

    The returned copy differs from *fragment* in one inert attribute: :data:`ASSET_MARKER` on the root element, naming the sidecar's URL. That is what lets :func:`link_externalized` replace a screenful of inlined SVG with a link in the Markdown rendition — the sidecar is written under a name of the caller's choosing, so without the stamp, matching an SVG in the document back to the file it came from would mean comparing content. Pass a single root element: the stamp lands on the first tag, and it is that one element the swap removes. Give it an ``aria-label`` (``figure_html``'s *aria_label*) and the link's text is that description — the same text a screen reader gets.

    The sidecar holds the fragment as authored, without the stamp: it is the figure, not a reference to itself.
    """
    publish = publish if publish is not None else current_publisher()
    if publish is None:
        return fragment
    leaf = name if PurePosixPath(name).suffix else f"{name}.html"
    url = publish.asset_url(fragment.encode(), name=leaf, serve=False)
    m = _ROOT_TAG.match(fragment)
    if m is None:
        log.warning("externalize_html: %r does not start with an element, so a render can't link it", leaf)
        return fragment
    return f'{fragment[: m.end()]} {ASSET_MARKER}="{html_escape(url)}"{fragment[m.end() :]}'


# The opening tag of an element carrying the stamp, with room for other attributes on
# either side of it.
_STAMPED_TAG = re.compile(rf'<(?P<tag>[a-zA-Z][\w.:-]*)\b[^>]*\s{ASSET_MARKER}="[^"]*"[^>]*>')
_ATTR = re.compile(r'([\w-]+)="([^"]*)"')
_IMAGE_SUFFIXES = {".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _element_end(text: str, start: int, tag: str) -> int:
    """The index just past the ``</tag>`` that closes the element opening at *start* (nesting respected)."""
    open_re = re.compile(rf"<{tag}\b")
    close_re = re.compile(rf"</{tag}>")
    pos = text.index(">", start) + 1
    depth = 1
    while depth:
        nxt_open = open_re.search(text, pos)
        nxt_close = close_re.search(text, pos)
        if nxt_close is None:
            raise ValueError(f"unbalanced <{tag}> at {start}")
        if nxt_open and nxt_open.start() < nxt_close.start():
            depth += 1
            pos = nxt_open.end()
        else:
            depth -= 1
            pos = nxt_close.end()
    return pos


def link_externalized(text: str) -> str:
    """Replace each fragment that has a sidecar with a link to it, in place of its markup.

    A report inlines some figures as SVG rather than as an ``<img>`` — subline strips, the swatch table — because inlined markup participates in the page's CSS, which a referenced file can't. That is right for the page and wrong for its Markdown rendition: on a report with eight of them the path data is more than half the document, sitting between the paragraphs a reader is there for. :func:`externalize_html` has already written each one out as a plain file beside the PNGs and stamped its URL on the element (:data:`ASSET_MARKER`), so the whole element can go and a link stand for it.

    The link's text is the element's ``aria-label``, which is the description a screen reader gets and the closest thing these fragments have to alt text; failing that, the sidecar's stem, and a warning, since an undescribed figure is as opaque to a reader of the page as to a reader of this. A sidecar that is itself an image (``.svg``) is written as an image; an ``.html`` one as a plain link, which is what it is.
    """
    out: list[str] = []
    pos = 0
    while (m := _STAMPED_TAG.search(text, pos)) is not None:
        out.append(text[pos : m.start()])
        attrs = dict(_ATTR.findall(m.group(0)))
        url = html_unescape(attrs[ASSET_MARKER])
        leaf = PurePosixPath(url.partition("?")[0]).name
        if label := attrs.get("aria-label"):
            alt = html_unescape(label)
        else:
            alt = PurePosixPath(leaf).stem
            log.warning("%s carries no aria-label; the Markdown links it as %r", leaf, alt)
        bang = "!" if PurePosixPath(leaf).suffix in _IMAGE_SUFFIXES else ""
        out.append(f"\n\n{bang}[{alt}]({url})\n\n")
        pos = m.end() if m.group(0).endswith("/>") else _element_end(text, m.start(), m.group("tag"))
    out.append(text[pos:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Publish: repoint a report's relative URLs at the bucket
# ---------------------------------------------------------------------------

# Matches the value of an ``src=`` / ``href=`` attribute, whether it sits in plain
# HTML (``src="…"``) or JSON-escaped inside a ``<script>`` blob
# (``src=\"…\"``) — hence the optional leading backslash and stopping at a backslash.
_URL_ATTR = re.compile(r'(?:src|href)\s*=\s*\\?["\']([^"\'\\]+)')

# A URL is "external/anchored" (not a relative path we'd resolve against a base) if it
# carries a scheme (``https:``, ``data:``, ``mailto:``…), is protocol-relative (``//``),
# or is a bare fragment (``#cell-id``).
_ANCHORED = re.compile(r"(?:[a-z][a-z0-9+.\-]*:|//|#)", re.IGNORECASE)


def relative_urls(html: str) -> list[str]:
    """Every relative ``src``/``href`` URL in *html* (escaped-in-JSON or not, in order)."""
    return [u for u in _URL_ATTR.findall(html) if u and not _ANCHORED.match(u)]


def stray_links(html: str, *, link: str = "_assets") -> list[str]:
    """Relative URLs that are *not* bundle-local — the ones a ``<base>`` would break.

    Bundle-local is a store asset (under *link*) or a rendition the page declares beside itself (:func:`alternates`, ``report.pdf``): both ride the bundle sync and resolve wherever the ``<base>`` points. What remains are author-written nav/source links (``./experiment.py``) that should be absolute. Returned sorted and de-duplicated so a build can resolve or warn on them.
    """
    prefix = f"{link}/"
    local = set(alternates(html).values())
    return sorted({u for u in relative_urls(html) if not u.startswith(prefix) and u not in local})


# One <img> tag, whole. Safe against the tag's own text: attribute values are
# HTML-escaped at authoring time (themed_figure_html html.escape()s them), so a
# literal ">" never appears inside one.
_IMG_TAG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)

# src / alt attribute values within one tag's text, in plain HTML or JSON-escaped inside
# a JSON blob (hence the optional backslash before each quote). src stops at a
# backslash — asset paths carry none. alt may contain JSON escapes (\uXXXX for non-ASCII)
# but never an escaped quote (a literal quote was HTML-escaped to &quot; before the JSON
# layer), so a backslash is consumed only when it escapes a non-quote character — which
# leaves the closing \" delimiter to end the match.
_IMG_SRC_ATTR = re.compile(r'(?<![\w-])src\s*=\s*\\?["\']([^"\'\\]+)', re.IGNORECASE)
_IMG_ALT_ATTR = re.compile(r'(?<![\w-])alt\s*=\s*\\?["\']((?:\\[^"\']|[^"\'\\])*)', re.IGNORECASE)
_IMG_WIDTH_ATTR = re.compile(r'(?<![\w-])width\s*=\s*\\?["\']?(\d+)', re.IGNORECASE)
_IMG_HEIGHT_ATTR = re.compile(r'(?<![\w-])height\s*=\s*\\?["\']?(\d+)', re.IGNORECASE)

_LIGHT_SUFFIX = "-light.png"
_DARK_SUFFIX = "-dark.png"


def _decode_attr(value: str) -> str:
    """An attribute value as authored: JSON-unescaped if it came from the session blob, then HTML-unescaped."""
    from html import unescape

    if "\\" in value:
        try:
            value = cast(str, json.loads(f'"{value}"'))
        except ValueError:
            pass
    return unescape(value)


@dataclass(frozen=True)
class ReportFigure:
    """One figure in a report's exported HTML, with its light/dark variants folded together.

    ``light`` and ``dark`` are the bundle-relative asset URLs (``_assets/<stem>-light.png``); ``dark`` is ``None`` for an unthemed image. ``alt`` is the figure's own alt text, as authored. ``width``/``height`` are the CSS-pixel display size the export stamped on the ``<img>`` (see :func:`mini.vis.figures.themed_figure_html`), or ``None`` when the tag carried none.
    """

    stem: str
    light: str
    dark: str | None = None
    alt: str = ""
    width: int | None = None
    height: int | None = None
    # The small copies :func:`write_thumbnails` made at export, when the bundle declares
    # them (:data:`THUMBNAIL_META`); ``None`` for a bundle exported before thumbnails.
    light_thumb: str | None = None
    dark_thumb: str | None = None


# The ``<meta>`` an export stamps into a bundle's ``<head>`` once :func:`write_thumbnails`
# has run; its ``content`` is the bundle-relative dir prefix (``_assets/thumbs/``) under
# which every asset-served figure has a same-named thumbnail. The build reads it from the
# one thing it fetches — the HTML — so a bundle without the tag (exported before
# thumbnails existed) degrades to full-size images rather than to broken ones.
THUMBNAIL_META = "mini-thumbnails"
THUMBNAIL_DIR = "thumbs"  # under the bundle's asset dir; the leaf names match their sources
THUMBNAIL_HEIGHT = 192  # px: the index strip shows ~72 CSS px tall, so this covers a 2× screen with margin

_THUMBNAIL_META_TAG = re.compile(
    rf"""<meta\s+name\s*=\s*["']{THUMBNAIL_META}["']\s+content\s*=\s*["']([^"']*)["']""", re.IGNORECASE
)


def _thumbnail_prefix(html: str) -> str | None:
    m = _THUMBNAIL_META_TAG.search(html)
    return m.group(1) if m else None


def report_figures(html: str, *, link: str = "_assets") -> list[ReportFigure]:
    """The asset-served figures in a report's exported HTML, in document order.

    A ``themed`` figure lands as two sibling ``<img>`` tags — ``<stem>-light.png`` and ``<stem>-dark.png``, one hidden by CSS — which fold into a single :class:`ReportFigure` keyed by the stem; an unpaired image keeps its filename stem and no ``dark``. The first alt text seen for a stem wins (the variants carry the same one). This is how the site build draws an index page's thumbnails from the HTML it already fetched, rather than listing the bucket: the HTML also knows the *narrative* order, which a directory listing doesn't.

    When the HTML carries the :data:`THUMBNAIL_META` tag, each figure's ``light_thumb``/``dark_thumb`` names the small copy under that prefix — same leaf, so the build derives the URL without listing anything.
    """
    prefix = f"{link}/"
    thumbs = _thumbnail_prefix(html)

    def thumb(src: str | None) -> str | None:
        return f"{thumbs}{src[len(prefix) :]}" if thumbs is not None and src else None

    # A JSON blob escapes angle brackets (< / >), so tags inside one
    # are invisible to an HTML-shaped regex until those are folded back. Quotes stay in
    # their \" form, which the attribute patterns already read.
    for esc, ch in (("\\u003C", "<"), ("\\u003c", "<"), ("\\u003E", ">"), ("\\u003e", ">")):
        html = html.replace(esc, ch)
    order: dict[str, dict[str, str | None]] = {}
    for tag in _IMG_TAG.findall(html):
        m = _IMG_SRC_ATTR.search(tag)
        if m is None or not m.group(1).startswith(prefix):
            continue
        src = m.group(1)
        leaf = src[len(prefix) :]
        if leaf.endswith(_LIGHT_SUFFIX):
            stem, role = leaf[: -len(_LIGHT_SUFFIX)], "light"
        elif leaf.endswith(_DARK_SUFFIX):
            stem, role = leaf[: -len(_DARK_SUFFIX)], "dark"
        else:
            stem, role = PurePosixPath(leaf).stem, "light"
        entry = order.setdefault(stem, {"light": None, "dark": None, "alt": None, "width": None, "height": None})
        entry[role] = entry[role] or src
        if entry["alt"] is None and (alt := _IMG_ALT_ATTR.search(tag)) is not None:
            entry["alt"] = _decode_attr(alt.group(1))
        # First tag seen wins, like alt: the export writes the light variant first, and
        # the pair's sizes differ by at most a pixel of tight-bbox rounding.
        if entry["width"] is None and (w := _IMG_WIDTH_ATTR.search(tag)) and (h := _IMG_HEIGHT_ATTR.search(tag)):
            entry["width"], entry["height"] = w.group(1), h.group(1)
    return [
        ReportFigure(
            stem,
            light=e["light"] or (e["dark"] or ""),
            dark=e["dark"],
            alt=e["alt"] or "",
            width=int(e["width"]) if e["width"] else None,
            height=int(e["height"]) if e["height"] else None,
            light_thumb=thumb(e["light"] or e["dark"]),
            dark_thumb=thumb(e["dark"]),
        )
        for stem, e in order.items()
        if e["light"] or e["dark"]
    ]


def _thumbnail_bytes(src: Path, height: int) -> bytes | None:
    """*src* scaled to *height* px tall, in its own format — or ``None`` when the original should ship as is (already that small, or not an image Pillow reads).

    A PNG is quantized to an 8-bit palette on the way out: a plot at thumbnail scale is a few colours on a flat background, and the palette cuts the file to roughly a quarter of the truecolour encoding for no visible loss. Other formats keep their encoder's defaults. Deterministic for unchanged input, so republishing an unchanged report writes the same bytes and stays a no-op commit.
    """
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(src) as im:
            fmt = im.format
            if im.height <= height:
                return None  # already small: ship the original rather than re-encode it
            im = im.convert("RGBA") if fmt == "PNG" else im.convert("RGB")
            im = im.resize((max(1, round(im.width * height / im.height)), height), Image.Resampling.LANCZOS)
            if fmt == "PNG":
                # Only the octree quantizer takes alpha, and it nudges it (opaque → 254);
                # an opaque figure — most of them — gets the better RGB quantizer instead.
                opaque = im.getchannel("A").getextrema() == (255, 255)
                im = im.convert("RGB").quantize(256) if opaque else im.quantize(256, method=Image.Quantize.FASTOCTREE)
            out = BytesIO()
            im.save(out, format=fmt, optimize=True)
    except UnidentifiedImageError, OSError:
        return None
    return out.getvalue()


def write_thumbnails(
    html: str, asset_dir: Path, *, link: str = "_assets", height: int = THUMBNAIL_HEIGHT
) -> tuple[str, list[str]]:
    """Write a thumbnail of every asset-served figure in *html* under ``asset_dir/thumbs/`` and declare them in the HTML.

    Returns the HTML with the :data:`THUMBNAIL_META` tag added, and the leaves written. Runs at export (``scripts/export_reports.py``), the one half of publishing that holds the figure bytes: the site build is read-only and fetches only the HTML, so it can't scale anything itself — it reads the tag and swaps each strip image for its small copy (:func:`report_figures`). The thumbnails ride the bundle sync like any other asset, so the index's images are pinned to the same revision as the report's.

    The tag promises a thumbnail for *every* figure the HTML references, since the build derives each URL from its source's leaf without listing the bucket. So a figure that can't be scaled — an SVG, a missing file, one already small enough — is copied through unchanged: that entry costs what it does today rather than 404ing.
    """
    import shutil

    prefix = f"{link}/"
    thumb_dir = asset_dir / THUMBNAIL_DIR
    written: list[str] = []
    for fig in report_figures(html, link=link):
        for src in (fig.light, fig.dark):
            if not src or not src.startswith(prefix):
                continue
            leaf = src[len(prefix) :]
            source, dest = asset_dir / leaf, thumb_dir / leaf
            if not source.is_file() or leaf in written:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if (data := _thumbnail_bytes(source, height)) is not None:
                dest.write_bytes(data)
            else:
                shutil.copyfile(source, dest)
            written.append(leaf)
    meta = f'<meta name="{THUMBNAIL_META}" content="{prefix}{THUMBNAIL_DIR}/" />'
    html = _THUMBNAIL_META_TAG.sub("", html)  # re-stamp rather than stack, if the HTML already carried one
    return re.sub(r"(<head[^>]*>)", lambda m: f"{m.group(1)}\n    {meta}", html, count=1), written


#: Stamped on every asset-served figure the build marks (:func:`mark_figures`, and the
#: index strip's thumbnails). It is what the lightbox script looks for, so a figure opts
#: in by carrying it rather than by the script guessing from a URL shape.
ZOOM_ATTR = "data-mini-zoom"

# An ``<img>`` tag in either spelling an export carries it in: plain markup, and
# JSON-escaped inside a script blob (``\u003Cimg … /\u003E``, attribute quotes
# as ``\"``). One pattern for both means one pass over the document — which is
# megabytes — and one place that knows the two spellings. Neither form holds a bare
# ``>`` inside a tag, so the body is "anything up to whichever terminator this form
# uses".
_ANY_IMG_TAG = re.compile(
    r"(?P<lt><|\\u003[Cc])img\b(?P<attrs>(?:(?!\\u003[Ee])[^>])*)(?P<gt>>|\\u003[Ee])", re.IGNORECASE
)
_LOADING_ATTR = re.compile(r"(?<![\w-])loading\s*=", re.IGNORECASE)


def mark_figures(html: str, *, link: str = "_assets") -> str:
    """Mark the asset-served figures in a report's HTML as deferrable and zoomable.

    Each matching ``<img>`` gains ``loading="lazy"`` — a report with twenty figures otherwise fetches all twenty PNGs on load, and a themed figure ships *both* variants even though CSS hides one — plus :data:`ZOOM_ATTR` and a ``tabindex``, which is how the lightbox (:func:`set_lightbox`) knows what to open and how a keyboard reaches it.

    Runs at build time beside :func:`set_lightbox` and friends, so it reaches reports published before any of this existed and a later change to the markup needs no re-export. Only figures under *link* are touched: an inline ``data:`` image is already downloaded, and an off-site one isn't ours to defer. Idempotent — a tag that already declares ``loading`` is left alone.
    """
    prefix = f"{link}/"

    def repl(m: re.Match) -> str:
        attrs = m["attrs"]
        src = _IMG_SRC_ATTR.search(attrs)
        if src is None or not src.group(1).startswith(prefix):
            return m[0]
        if _LOADING_ATTR.search(attrs) or ZOOM_ATTR in attrs:
            return m[0]
        q = '"' if m["lt"] == "<" else '\\"'  # inside the blob the tag's own quotes are escaped
        return f"{m['lt']}img loading={q}lazy{q} tabindex={q}0{q} {ZOOM_ATTR}{attrs}{m['gt']}"

    return _ANY_IMG_TAG.sub(repl, html)


#: Markup a heading carries that GitHub renders away before slugging: inline HTML, and
#: a link's target (its text stays).
_SLUG_HTML_TAG = re.compile(r"<[^>]+>")
_SLUG_LINK = re.compile(r"\[(?P<text>[^\]]*)\]\([^)]*\)")

#: Emphasis markers. `*` only, never `_`: CommonMark doesn't open emphasis on an intra-word
#: underscore, so the underscores in this repo's headings are all identifiers —
#: `test_local_apparatus_concurrent`, `__init__.py` — and reading a pair of them as emphasis
#: would eat the characters between, slugging the first `testlocalapparatus_concurrent` and
#: never matching a real link.
_SLUG_EMPHASIS = re.compile(r"\*{1,3}(?P<text>[^*]+?)\*{1,3}")

#: What survives: letters, digits, spaces, hyphens, underscores. Spaces then become hyphens.
_SLUG_STRIP = re.compile(r"[^\w\- ]", re.UNICODE)


def github_slug(heading: str) -> str:
    """A heading's GitHub anchor: strip markup, lowercase, drop punctuation, spaces to hyphens.

    The one definition of a heading anchor for this repo, shared by ``scripts/check_md_links.py`` (which validates a ``#fragment`` against it) and ``scripts/build_site.py`` (which hands it to Python-Markdown's ``toc`` so the published page carries the same id GitHub would give it). Python-Markdown's own slugify collapses a run of separators, so "Provenance & cost" would be ``provenance-cost`` there and ``provenance--cost`` here — the removed ``&`` leaves two spaces, and GitHub turns both into hyphens. A link that resolves in one place has to resolve in the other, so neither may drift.

    Repeats are the one case the two still number differently (``-1`` on GitHub, ``_1`` in ``toc``); ``check_md_links`` reports a duplicate heading under ``docs/`` rather than letting it render.
    """
    text = _SLUG_HTML_TAG.sub("", heading)
    text = _SLUG_LINK.sub(lambda m: m["text"], text)
    text = _SLUG_EMPHASIS.sub(lambda m: m["text"], text)
    text = text.replace("`", "")
    # NFC first, so a combining accent and its precomposed form slug alike.
    text = unicodedata.normalize("NFC", text).lower()
    return _SLUG_STRIP.sub("", text).replace(" ", "-")


def rewrite_links(html: str, mapping: dict[str, str]) -> str:
    r"""Replace each relative URL in *mapping* (token → absolute target) throughout *html*.

    Targets the URL only where it sits as a quoted attribute value, in both plain (``href="../a/report.py"``) and JSON-escaped (``href=\"../a/report.py\"``) form, and either quote style — the same shapes :func:`relative_urls` matches. The replacement is an absolute URL (no quotes/backslashes of its own), so it's valid in either context; anchoring on the surrounding quotes keeps a short token from matching inside an unrelated string.
    """
    for token, target in mapping.items():
        for q in ('"', "'"):
            html = html.replace(f"{q}{token}{q}", f"{q}{target}{q}")  # plain
            html = html.replace(f"\\{q}{token}\\{q}", f"\\{q}{target}\\{q}")  # escaped-in-JSON
    return html


_FRAGMENT_HREF = re.compile(r"""(href\s*=\s*)(["'])#""")


def insert_base(html: str, href: str, *, page_url: str | None = None) -> str:
    """Insert a single ``<base href>`` as the first thing in ``<head>``, and pin the page's own fragment links to *page_url*.

    Placed before any resource reference so it governs all of them. Idempotent enough for a build step: it rewrites the first ``<head>`` only.

    A ``<base>`` repoints *every* relative URL, and a bare ``#section`` is one: the browser resolves it against the base, so a footnote, a heading permalink, or a table-of-contents entry would leave for the bucket. With *page_url* (the page's own published URL), each ``href="#…"`` becomes ``href="<page_url>#…"``, which is the same document again. Without it the fragments are left alone, which is the broken form; the caller should know its URL.
    """
    if page_url is not None:
        html = _FRAGMENT_HREF.sub(lambda m: f"{m.group(1)}{m.group(2)}{page_url}#", html)
    return re.sub(r"(<head[^>]*>)", lambda m: f'{m.group(1)}\n    <base href="{href}" />', html, count=1)


# The document's real ``<body>`` open tag. A bare search for ``<body`` is not enough: the
# text also occurs in prose, and a CSS comment in the baked ``report.css`` once said
# "an explicit theme on <body> wins", which put the nav chip, the provenance chip and the
# flash guard inside a ``<style>`` element in the head, where the browser never renders
# them. The tag that follows ``</head>`` is the one; the bare match is the fallback for a
# fragment with no head at all.
_BODY_OPEN = re.compile(r"</head>\s*(<body\b[^>]*>)", re.IGNORECASE)
_BODY_OPEN_BARE = re.compile(r"(<body\b[^>]*>)", re.IGNORECASE)


def _after_body_open(html: str, snippet: str) -> str:
    """*html* with *snippet* inserted as the first thing in the document body."""
    m = _BODY_OPEN.search(html) or _BODY_OPEN_BARE.search(html)
    if m is None:
        return html
    return f"{html[: m.end(1)]}\n    {snippet}{html[m.end(1) :]}"


# The lightbox's own styling. ``Canvas``/``CanvasText`` are the UA's theme-aware system
# colors — the same ones the nav and provenance chips use — so the panel behind the
# figure is the page's own background in either scheme, which is what a figure PNG's
# transparent background needs to read correctly. Opening the file in a tab instead would
# paint it on the browser's white canvas, wrong in dark mode; this is the reason the
# index strip had no link to the full-size image until now.
_LIGHTBOX_CSS = """
[data-mini-zoom]{cursor:zoom-in}
dialog.mini-lightbox{border:0;padding:0;margin:auto;max-width:96vw;max-height:96vh;
  background:Canvas;color:CanvasText;border-radius:.5rem;box-shadow:0 1rem 3rem rgb(0 0 0/.45)}
dialog.mini-lightbox::backdrop{background:rgb(0 0 0/.62);-webkit-backdrop-filter:blur(3px);backdrop-filter:blur(3px)}
dialog.mini-lightbox img{display:block;margin:0 auto;max-width:96vw;max-height:84vh;width:auto;height:auto;
  transition:filter .18s ease-out,opacity .18s ease-out,width .18s ease-out}
dialog.mini-lightbox img.mini-lightbox-loading{filter:blur(4px) grayscale(.35);opacity:.55}
@media (prefers-reduced-motion:reduce){dialog.mini-lightbox img{transition:none}}
dialog.mini-lightbox figcaption{margin:0;padding:.5rem .75rem;max-width:70ch;
  font:italic .8125rem/1.5 system-ui,sans-serif;color:color-mix(in srgb,CanvasText 70%,transparent)}
.mini-lightbox-close{position:absolute;top:.35rem;right:.35rem;width:1.9rem;height:1.9rem;
  font:1.25rem/1 system-ui,sans-serif;cursor:pointer;color:CanvasText;border-radius:50%;
  background:color-mix(in srgb,Canvas 70%,transparent);
  border:1px solid color-mix(in srgb,CanvasText 20%,transparent)}
@media print{dialog.mini-lightbox{display:none}}
"""

# A ``<dialog>`` opened with ``showModal`` renders in the browser's *top layer*, above
# every stacking context on the page — which is why the overlay needs none of the
# z-index arithmetic the nav and provenance chips do.
# It also brings Escape-to-close, the focus trap, and inertness of the page behind it for
# free. Listeners are delegated from ``document``, so figures added after this script
# runs are covered without re-binding.
_LIGHTBOX_JS = r"""
(function(){
  if(!window.HTMLDialogElement) return;  // no dialog: the figures simply stay static
  var ZOOM='[__ZOOM__]', dlg, seq=0;
  function chrome(){
    if(dlg) return dlg;
    dlg=document.createElement('dialog');
    dlg.className='mini-lightbox';
    dlg.innerHTML='<button type="button" class="mini-lightbox-close" aria-label="Close">×</button>'+
      '<figure style="margin:0"><img alt=""><figcaption></figcaption></figure>';
    dlg.addEventListener('click',function(ev){
      // The dialog box is exactly the panel, so a click reported against the dialog
      // itself landed on the backdrop around it.
      if(ev.target===dlg||ev.target.closest('.mini-lightbox-close')) dlg.close();
    });
    document.body.appendChild(dlg);
    return dlg;
  }
  function fullSrc(img){
    var light=img.getAttribute('data-mini-full'), dark=img.getAttribute('data-mini-full-dark');
    if(light||dark){
      // An index thumbnail names its full-size counterparts. The index is a plain page
      // with no theme of its own, so the device preference is the whole of the answer.
      return (dark&&matchMedia('(prefers-color-scheme: dark)').matches&&dark)||light||dark;
    }
    // A themed report figure is a light/dark pair of siblings with one hidden by CSS.
    // Whichever is on screen is the variant the page settled on, however it decided.
    var pair=(img.parentElement||img).querySelectorAll('img.mini-themed-img-light,img.mini-themed-img-dark');
    for(var i=0;i<pair.length;i++) if(pair[i].offsetParent) return pair[i].src;
    return img.src;
  }
  function want(img){
    // The size the report asked for, stamped by the export on the figure's tag and on the
    // index thumbnail made from it: the figure's *physical* size (see mini.vis.figures), which
    // for a plot saved at 2x is half its pixels. That is the size it is meant to be drawn
    // at and the size it looks sharp at on a dense screen, so it is the panel's ceiling
    // as well as its shape — and it is in the markup before the panel has any bytes.
    return {w:+img.getAttribute('width')||0, h:+img.getAttribute('height')||0};
  }
  function ratio(img){
    var d=want(img), w=d.w, h=d.h;
    if(!(w>0&&h>0)){ w=img.naturalWidth; h=img.naturalHeight; }  // unstamped: what it measures
    return (w>0&&h>0) ? w/h : 0;
  }
  function size(ar, px){
    // As large as the viewport allows, up to the figure's own size. An unstamped figure
    // has only its pixels to go on, and not until it has them — pass 0 and the fit stands.
    if(!ar) return '';
    return 'min(96vw,calc(84vh * '+ar.toFixed(4)+(px?'),'+px+'px':')')+')';
  }
  function open(img){
    var d=chrome(), big=d.querySelector('img'), cap=d.querySelector('figcaption'),
        fig=img.closest('figure'), own=fig&&fig.querySelector('figcaption'),
        full=fullSrc(img), shown=img.currentSrc||img.src, ar=ratio(img), w=want(img).w;
    // Fix the box before any bytes arrive: an image that hasn't loaded has no intrinsic
    // size, so without this the panel opens flat and jumps open when the figure lands.
    // A stamped figure opens at its final size; an unstamped one at the right shape,
    // narrowing to fit its pixels once those arrive.
    big.style.aspectRatio = ar ? ar.toFixed(4) : '';
    big.style.width = size(ar, w || (shown===full ? img.naturalWidth : 0));
    big.alt=img.alt||'';
    cap.textContent=((own&&own.textContent)||img.alt||'').trim();
    cap.hidden=!cap.textContent;
    // Start from the image the page is already showing — on the index a thumbnail the
    // browser has cached, on a report the full-size figure itself — so the panel paints
    // this frame rather than holding the figure opened before it. Blown up from a
    // thumbnail it is soft, so it reads as a placeholder until the real one arrives.
    var turn = ++seq;
    big.src=shown;
    big.classList.toggle('mini-lightbox-loading', shown!==full);
    d.showModal();
    if(shown===full) return;
    var pre=new Image();
    // Swapping to a decoded image paints without an empty frame in between. A second
    // click while this one loads takes the panel, so a stale arrival is dropped.
    pre.onload=pre.onerror=function(){
      if(turn!==seq) return;
      if(pre.naturalWidth){ big.src=full; big.style.width=size(ar, w || pre.naturalWidth); }
      big.classList.remove('mini-lightbox-loading');
    };
    pre.src=full;
  }
  function target(ev){
    return ev.target.closest?ev.target.closest(ZOOM):null;
  }
  document.addEventListener('click',function(ev){
    if(ev.button||ev.metaKey||ev.ctrlKey||ev.shiftKey||ev.altKey) return;
    var img=target(ev);
    if(img){ ev.preventDefault(); open(img); }
  });
  document.addEventListener('keydown',function(ev){
    if(ev.key!=='Enter'&&ev.key!==' ') return;
    var img=target(ev);
    if(img){ ev.preventDefault(); open(img); }
  });
  // A deferred figure (mark_figures) that has not scrolled into view has no pixels yet,
  // and would print as a blank box. Paper has no viewport, so load them all first.
  window.addEventListener('beforeprint',function(){
    document.querySelectorAll('img[loading=lazy]').forEach(function(img){ img.loading='eager'; });
  });
})();
""".replace("__ZOOM__", ZOOM_ATTR)


def lightbox_chrome() -> str:
    """The ``<style>`` and ``<script>`` that make a :data:`ZOOM_ATTR` figure open full-size in an overlay.

    One snippet, inlined by both page builders — the report pages through :func:`set_lightbox`, the Markdown pages by ``scripts/build_site.py`` — rather than a file the site links, because externalize mode puts a ``<base href>`` at the bucket that would repoint a relative stylesheet URL, and because the two would otherwise drift apart.
    """
    return f"<style>{_LIGHTBOX_CSS.strip()}</style>\n<script>{_LIGHTBOX_JS.strip()}</script>"


def set_lightbox(html: str) -> str:
    """Give a published report's figures a click-to-enlarge overlay.

    A figure on the page is sized to the reading column, and the PNG behind it is often several times that wide; clicking one now opens it at full size over a dimmed page, captioned, and closes on Escape, on the backdrop, or on the button. It stays on the page throughout — no navigation — which is what lets the overlay carry a themed background, so a figure's transparent PNG reads right in dark mode as well as light.

    Pairs with :func:`mark_figures`, which is what says *which* images are figures. A no-op on a page with no ``</head>``.
    """
    return re.sub(r"(</head>)", lambda m: f"    {lightbox_chrome()}\n{m.group(1)}", html, count=1)


def set_report_styles(html: str, css: str) -> str:
    """Inline the shared report stylesheet (*css*) as the last thing in ``<head>``.

    The reports carry the same sheet two ways. The exporter bakes it into each bundle (so it ships in the raw bundle and the PDF print sees it); this re-inlines the *current* source at build time, landing after that baked copy — so editing ``docs/report.css`` restyles every published report with no re-export. It's inlined, not ``<link>``ed, because externalize mode inserts a ``<base href>`` at the bucket that would repoint a relative stylesheet URL (and inlining works offline too). A no-op on a page with no ``</head>`` to match, or when *css* is empty. Apply it last, so report rules win any specificity tie.
    """
    if not css.strip():
        return html
    style = f"<style>\n{css.strip()}\n    </style>"
    return re.sub(r"(</head>)", lambda m: f"    {style}\n{m.group(1)}", html, count=1)


# Our nav is *absolutely* positioned, not in normal flow, so it scrolls away with the
# document (it settles at the top of the page as a header rather than shadowing the
# content the whole way down). Pinned top-left; the content column gets matching top
# padding (:data:`_BANNER_CLEARANCE`) so the report's title isn't tucked under it. ``Canvas``/``CanvasText`` are the UA's
# theme-aware system colors (the export declares ``color-scheme``, so they track the
# device theme); a blurred translucent backdrop keeps it legible where it does overlap.
_BANNER_STYLE = (
    "position:absolute;top:.5rem;left:.5rem;z-index:2147483647;"
    "display:flex;gap:.75rem;align-items:center;"
    "padding:.3rem .65rem;font-size:.8125rem;line-height:1.4;"
    "font-family:system-ui,sans-serif;border-radius:.375rem;"
    "background:color-mix(in srgb, Canvas 80%, transparent);"
    "border:1px solid color-mix(in srgb, CanvasText 18%, transparent);"
    "-webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px);"
)
_BANNER_LINK = "color:inherit;text-decoration:underline"

# The nav is out of flow, so it reserves no space; without this the report's first line
# (its title) would sit under it at the top of the page. A little top padding on the
# content column (``main.lit``, a literate script's page) drops the whole report clear
# of the chip.
_BANNER_CLEARANCE = "main.lit{padding-top:3rem}"


# The same document in another format, declared in the ``<head>`` so a reader (a crawler, an
# agent fetching the page, a browser extension) can find it without parsing the body:
# ``<link rel="alternate" type="<media type>" href="…">`` is the HTML mechanism for exactly
# this (RFC 8288 ``alternate``, with ``type`` to say which format). The export stamps one
# per rendition it wrote beside ``index.html``; the build reads them to link each in the
# nav chip. Relative hrefs, so the page's ``<base>`` sends them wherever the bundle is.
PDF_LEAF = "report.pdf"  # printed at export by mini.report_print, beside index.html
PDF_TYPE = "application/pdf"
MD_LEAF = "index.md"  # the woven Markdown a literate script's export writes beside its HTML
MD_TYPE = "text/markdown"
_ALTERNATE_TAG = re.compile(r'\s*<link rel="alternate" type="([^"]*)" href="([^"]*)"\s*/?>', re.IGNORECASE)


def set_alternate(html: str, *, type: str, href: str) -> str:
    """Declare that this page is also available as *type* at *href* (a ``<link rel="alternate">`` in the head).

    One tag per media type: re-stamping the same type replaces the earlier tag rather than stacking. Run at export, on the bundle's HTML, since that is the half that knows what it wrote.
    """
    html = _ALTERNATE_TAG.sub(lambda m: "" if m.group(1) == type else m.group(0), html)
    tag = f'<link rel="alternate" type="{html_escape(type)}" href="{html_escape(href)}" />'
    return re.sub(r"(<head[^>]*>)", lambda m: f"{m.group(1)}\n    {tag}", html, count=1)


def alternates(html: str) -> dict[str, str]:
    """The alternate renditions a page declares, media type → href (see :func:`set_alternate`)."""
    return {m.group(1): m.group(2) for m in _ALTERNATE_TAG.finditer(html)}


def set_banner(
    html: str, *, index_url: str | None = None, source_url: str | None = None, pdf_url: str | None = None
) -> str:
    """Give a published report a floating nav — back to the index, out to the source, and the PDF.

    A small chip (``← Index`` · ``Source`` · ``PDF``) pinned top-left. It's absolutely positioned and scrolls away with the page; the content column is padded down so the title clears it. A link is omitted when its URL is ``None``; a no-op if none is given. The chip is hidden in print, so the PDF never links to itself.
    """
    if index_url is None and source_url is None and pdf_url is None:
        return html

    def link(href: str, label: str) -> str:
        return f'<a href="{href}" style="{_BANNER_LINK}">{label}</a>'

    entries = ((index_url, "&larr; Index"), (source_url, "Source"), (pdf_url, "PDF"))
    links = [link(url, label) for url, label in entries if url]
    bar = f'<nav data-mini-banner style="{_BANNER_STYLE}">{"".join(links)}</nav>'

    html = re.sub(
        r"(</head>)",
        lambda m: (
            f"    <style>{_BANNER_CLEARANCE}\n    @media print{{[data-mini-banner]{{display:none}}}}</style>\n{m.group(1)}"
        ),
        html,
        count=1,
    )
    return _after_body_open(html, bar)


# The provenance chip mirrors the nav's mechanics (absolute, UA system colors, blurred
# backdrop) but sits bottom-left and folds away behind a
# ``<details>`` — provenance should be *findable*, not competing with the report's
# content. Absolute (not fixed) so it too scrolls with the page, coming to rest at the
# foot of the report; the content column's own bottom padding keeps text clear of it.
_PROVENANCE_STYLE = (
    "position:absolute;bottom:.5rem;left:.5rem;z-index:2147483647;"
    "max-width:min(30rem,90vw);"
    "padding:.3rem .65rem;font-size:.75rem;line-height:1.5;"
    "font-family:system-ui,sans-serif;border-radius:.375rem;"
    "background:color-mix(in srgb, Canvas 85%, transparent);"
    "border:1px solid color-mix(in srgb, CanvasText 18%, transparent);"
    "-webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px);"
)
_PROVENANCE_DIM = "color:color-mix(in srgb, CanvasText 60%, transparent)"


def _provenance_entries(refs: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    """Fold ref → producer down to one entry per producing experiment (sorted).

    Unattributed refs (written before producer stamping, or outside a run) are dropped — the footer only makes claims it has evidence for.
    """
    by_exp: dict[str, dict[str, Any]] = {}
    for name, producer in sorted(refs.items()):
        if not producer or not producer.get("experiment"):
            continue
        entry = by_exp.setdefault(producer["experiment"], {**producer, "refs": []})
        entry["refs"].append(name)
    return [by_exp[k] for k in sorted(by_exp)]


def set_provenance(html: str, refs: dict[str, dict[str, Any] | None]) -> str:
    """Give a published report a folded data-provenance footer.

    *refs* is the bundle's provenance sidecar content (ref name → the producer stamped at ``set_ref`` time). Each producing experiment gets one line — name, code state, run date — with the resolved ref names beneath it, inside a ``<details>`` chip pinned bottom-left (scrolling to rest at the foot of the report). A report whose refs carry no producer (or that read no refs at all) is left untouched. Content is derived only from the store's refs, so re-exporting unchanged data injects the same footer.
    """
    entries = _provenance_entries(refs)
    if not entries:
        return html

    def line(e: dict[str, Any]) -> str:
        code = e.get("git_describe") or (e.get("git_sha") or "")[:12]
        bits = [f"<strong>{e['experiment']}</strong>"]
        if code:
            bits.append(f"<code>{code}</code>{' (dirty)' if e.get('git_dirty') else ''}")
        if run_at := e.get("run_at"):
            bits.append(f"run {str(run_at)[:10]}")
        via = f'<div style="{_PROVENANCE_DIM}">via {", ".join(e["refs"])}</div>'
        return f"<div>{' · '.join(bits)}{via}</div>"

    chip = (
        f'<details data-mini-provenance style="{_PROVENANCE_STYLE}">'
        f'<summary style="cursor:pointer;{_PROVENANCE_DIM}">Data provenance</summary>'
        f"{''.join(line(e) for e in entries)}"
        "</details>"
    )
    html = re.sub(
        r"(</head>)",
        lambda m: f"    <style>@media print{{[data-mini-provenance]{{display:none}}}}</style>\n{m.group(1)}",
        html,
        count=1,
    )
    return _after_body_open(html, chip)

"""
Report bundles: produce a report's assets as relative URLs, then repoint them.

A report is a **bundle** — one Marimo HTML document plus its heavy assets (figures,
data blobs). The two halves of the bundle protocol both live here:

**Produce.** A :class:`Publisher` writes each asset out as a file beside the exported
HTML and hands back a *relative* URL like ``_assets/<name>.png``. The path is the
asset's readable name (so a browser saving it suggests a sensible filename — the URL's
last segment, since the bucket sets no ``Content-Disposition``), and the name *is* the
key, so a re-render overwrites in place and the URL stays stable. ``themed`` figures
externalize through a publisher when one is set; :meth:`Publisher.asset_url` is the
general verb for any blob.

**Publish.** That same HTML is consumed two ways:

- **opened locally**, the relative URL resolves to the co-located ``_assets/`` files;
- **served from Pages**, we want it to resolve to the assets we uploaded to the HF
  bucket instead.

The bridge is a single ``<base href>`` in the ``<head>`` (:func:`insert_base`): it
sets the document base that *every* relative URL resolves against, so one inserted
tag repoints the whole report's assets at the bucket — no per-URL rewriting, and it
works for the data URIs buried in Marimo's session JSON and a relative ``fetch()``
alike.

The catch is that ``<base>`` is document-global, so an author-written relative *link*
(a markdown ``[src](./experiment.py)``) would be repointed too — and 404 against the
bucket. :func:`stray_links` finds those at build time; :func:`rewrite_links` turns
them into absolute targets (their rendered page, or their source) so they survive the
base. The convention is *the only relative URLs left in a report are store assets*.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

__all__ = [
    "Publisher",
    "report_bundle",
    "export_key",
    "export_dir",
    "is_report_notebook",
    "report_notebooks",
    "SOURCE_ONLY_MARKER",
    "use_publisher",
    "current_publisher",
    "relative_urls",
    "stray_links",
    "rewrite_links",
    "insert_base",
    "set_theme",
    "set_banner",
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


@dataclass(frozen=True)
class Publisher:
    """Writes a report's heavy assets out as files beside the exported HTML,
    referenced by a **relative** URL.

    Each blob is written under ``asset_dir`` (the report's bundle ``_assets/`` —
    :func:`report_bundle`) at its readable *name*. The name *is* the key, so the URL is stable across
    re-exports — a re-render overwrites in place rather than piling up a new
    content-addressed copy each time (which is what kept the bucket accumulating
    orphans). The name is also what a browser "Save as" suggests (it derives the
    filename from the URL's last segment, the bucket setting no
    ``Content-Disposition``). The reference is ``<link>/<name>``; because it's
    relative, the same HTML resolves to the local files when opened off disk and to
    the HF bucket when published (a single ``<base href>`` is inserted at build time
    — see ``scripts/build_site.py``).
    """

    asset_dir: Path
    link: str = "_assets"
    # name -> sha of what we wrote under it this export, so a second *different*
    # blob under the same name is caught rather than silently clobbering.
    _written: dict[str, str] = field(default_factory=dict, compare=False, repr=False)

    def asset_url(self, data: bytes | Path, *, name: str) -> str:
        """Write *data* (bytes or a file) as ``<name>`` and return its relative URL.

        The asset is keyed by its readable *name* (carry the extension — it sets the
        served media type), so the URL is stable and a re-render overwrites in place.
        Two *different* blobs written under the same name in one report is an authoring
        bug (give each figure a distinct ``name=``), so it raises rather than clobber.
        """
        blob = bytes(data) if isinstance(data, (bytes, bytearray)) else Path(data).read_bytes()
        leaf = _safe_leaf(name)
        sha = hashlib.sha256(blob).hexdigest()
        if (prev := self._written.get(leaf)) is not None and prev != sha:
            raise ValueError(
                f"two different assets written as {leaf!r} in one report — pass a distinct "
                "name= to disambiguate (the asset name is the stable URL now, with no content hash)"
            )
        self._written[leaf] = sha
        dest = self.asset_dir / leaf
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{leaf}.tmp")
        tmp.write_bytes(blob)
        tmp.replace(dest)  # atomic + overwrite-in-place: a re-render replaces, never piles up
        return f"{self.link}/{leaf}"


def _project_root(start: Path) -> Path:
    """The project root (nearest ``pyproject.toml`` / ``.git``) walking up from *start*.

    Anchored at the *path*, not the cwd, so it's stable during ``marimo export`` (which
    may run from anywhere) — ``__file__`` is absolute there.
    """
    start = start.resolve()
    for d in (start, *start.parents):
        if any((d / m).exists() for m in _ROOT_MARKERS):
            return d
    return start.parent


def export_key(notebook_file: str | Path) -> str:
    """The docs-relative, suffix-less key naming a report's self-contained bundle.

    ``docs/gpt.py`` → ``gpt``; ``docs/gpt-sweep/report.py`` → ``gpt-sweep``. A report
    named ``report.py`` takes its *directory* as the key, so the common one-experiment,
    one-report split publishes at ``gpt-sweep/`` rather than the redundant
    ``gpt-sweep/report/``. A second report alongside it keeps its own stem
    (``docs/foo/aside.py`` → ``foo/aside``), so the convention extends to multiple
    reports per experiment without collision. The key names the report's on-disk export
    dir *and* its ``exports/<key>/`` prefix on the bucket, and (served as ``index.html``)
    its URL ``<key>/`` — so each report is one independently syncable bundle.
    """
    p = Path(notebook_file).resolve()
    docs = _project_root(p) / "docs"
    try:
        rel = p.relative_to(docs)
    except ValueError:
        rel = Path(p.name)
    key = rel.with_suffix("")
    if key.name == "report" and key.parent != Path("."):
        key = key.parent  # a directory's canonical report drops the redundant /report
    return key.as_posix()


def export_dir(notebook_file: str | Path) -> Path:
    """The local (gitignored) dir holding a report's exported ``index.html`` + ``_assets/``.

    ``<root>/.mini/exports/<key>/`` — the unit that mirrors to bucket ``exports/<key>/``.
    Kept under ``.mini`` (already gitignored) so exported HTML never enters Git.
    """
    p = Path(notebook_file).resolve()
    return _project_root(p) / ".mini" / "exports" / export_key(p)


# A docs notebook carrying this marker is a source-only *example*, not a rendered
# report: the build skips it (never runs its inline compute) and links to it resolve
# to its GitHub source instead of a site page. For notebooks that don't fit the
# read-from-store report model — e.g. ``docs/gpt.py`` trains inline on every run, so
# exporting it would re-run the whole experiment. Put it in a cell the notebook tool
# preserves (e.g. the setup block), since the text is matched literally.
SOURCE_ONLY_MARKER = "mini:source-only"


def is_report_notebook(path: str | Path) -> bool:
    """Whether *path* is a Marimo report the site renders.

    A report is a ``.py`` that declares ``marimo.App(`` and is *not* flagged
    ``# mini:source-only`` (:data:`SOURCE_ONLY_MARKER`): the marker opts a notebook out
    of the published set, so the build neither runs nor renders it and links to it fall
    back to its GitHub source. The notebooks are the only source of truth for the report
    set — a report is on the site iff its ``.py`` is in the repo and its bundle is synced.
    """
    p = Path(path)
    if p.suffix != ".py":
        return False
    try:
        text = p.read_text("utf-8", errors="ignore")
    except OSError:
        return False
    return "marimo.App(" in text and SOURCE_ONLY_MARKER not in text


def report_notebooks(docs: str | Path) -> list[Path]:
    """Every Marimo report notebook under *docs* (sorted); see :func:`is_report_notebook`."""
    return sorted(p for p in Path(docs).rglob("*.py") if is_report_notebook(p))


def report_bundle(notebook_file: str | Path, *, link: str = "_assets") -> Publisher:
    """A :class:`Publisher` writing assets beside a report's exported HTML.

    A report exports to its own self-contained dir :func:`export_dir` (HTML as
    ``index.html``, assets under ``_assets/``); this points the publisher at that dir's
    ``_assets/`` so the relative ``_assets/<name>`` URL resolves next to the HTML. Call
    it from the report's setup cell with ``__file__``::

        use_publisher(report_bundle(__file__))
    """
    return Publisher(asset_dir=export_dir(notebook_file) / link, link=link)


_default_publisher: Publisher | None = None


def use_publisher(publisher: Publisher | None) -> Publisher | None:
    """Set the report-wide default publisher; call once in a report's setup cell.

    Every ``@themed`` figure then externalizes through it with no per-figure argument.
    Pass a :class:`Publisher` (usually from :func:`report_bundle`), or ``None`` to clear
    it (figures inline as self-contained ``data:`` URIs). Returns it, e.g. to call
    :meth:`~Publisher.asset_url` for a data blob.
    """
    global _default_publisher
    _default_publisher = publisher
    return publisher


def current_publisher() -> Publisher | None:
    """The report-wide default publisher set by :func:`use_publisher` (or ``None``)."""
    return _default_publisher


# ---------------------------------------------------------------------------
# Publish: repoint a report's relative URLs at the bucket
# ---------------------------------------------------------------------------

# Matches the value of an ``src=`` / ``href=`` attribute, whether it sits in plain
# HTML (``src="…"``) or JSON-escaped inside Marimo's ``<script>`` session blob
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
    """Relative URLs that are *not* store assets — the ones a ``<base>`` would break.

    These are author-written nav/source links (``./experiment.py``) that should be
    absolute. Returned sorted and de-duplicated so a build can resolve or warn on them.
    """
    prefix = f"{link}/"
    return sorted({u for u in relative_urls(html) if not u.startswith(prefix)})


def rewrite_links(html: str, mapping: dict[str, str]) -> str:
    r"""Replace each relative URL in *mapping* (token → absolute target) throughout *html*.

    Targets the URL only where it sits as a quoted attribute value, in both plain
    (``href="../a/report.py"``) and JSON-escaped (``href=\"../a/report.py\"``) form,
    and either quote style — the same shapes :func:`relative_urls` matches. The
    replacement is an absolute URL (no quotes/backslashes of its own), so it's valid in
    either context; anchoring on the surrounding quotes keeps a short token from
    matching inside an unrelated string.
    """
    for token, target in mapping.items():
        for q in ('"', "'"):
            html = html.replace(f"{q}{token}{q}", f"{q}{target}{q}")  # plain
            html = html.replace(f"\\{q}{token}\\{q}", f"\\{q}{target}\\{q}")  # escaped-in-JSON
    return html


def insert_base(html: str, href: str) -> str:
    """Insert a single ``<base href>`` as the first thing in ``<head>``.

    Placed before any resource reference so it governs all of them. Idempotent enough
    for a build step: it rewrites the first ``<head>`` only.
    """
    return re.sub(r"(<head[^>]*>)", lambda m: f'{m.group(1)}\n    <base href="{href}" />', html, count=1)


# The ``display.theme`` inside Marimo's frozen mount config. The block is flat JSON
# (no nested objects), so ``[^{}]*?`` stays within it; ``count=1`` guards the rest.
_DISPLAY_THEME = re.compile(r'("display"\s*:\s*\{[^{}]*?"theme"\s*:\s*")(?:light|dark|system)(")')

# What the document declares to the browser, so the UA paints its chrome (the canvas
# behind the page, scrollbars, form controls) in the right scheme from the very first
# paint — before any stylesheet or script runs.
_COLOR_SCHEME = {"system": "light dark", "light": "light", "dark": "dark"}

# Runs synchronously as the first thing in <body> — before first paint, since Marimo's
# stylesheets are render-blocking and already loaded by then. It sets the same body
# markup Marimo applies (class="<t> <t>-theme" data-theme="<t>"), so the page paints in
# the device theme straight away instead of flashing light and correcting once Marimo's
# bundle mounts. Marimo recomputes the same value for a ``system`` config, so its later
# take-over is a no-op (no second repaint).
_FLASH_GUARD = (
    "<script>"
    "(function(){"
    'var t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";'
    'document.body.classList.add(t,t+"-theme");'
    "document.body.dataset.theme=t;"
    "})();"
    "</script>"
)


def set_theme(html: str, theme: str = "system") -> str:
    """Rewrite a Marimo export's theme so a published report follows the device, flicker-free.

    Marimo bakes the *exporting* machine's ``display.theme`` into the mount config and a
    bit of JS applies it on load (``<body class="light light-theme" data-theme="light">``).
    For a report served to other people that hard-codes one author's preference; rewriting
    it to ``system`` makes the frontend honor the visitor's ``prefers-color-scheme``.

    Because that JS only runs once Marimo's bundle mounts, the page would otherwise paint
    light first and flip — so for ``system`` we also declare ``<meta name="color-scheme">``
    (UA chrome) and inject a tiny blocking :data:`_FLASH_GUARD` (the content) to get the
    right theme on the first paint. A no-op if no theme is present (a non-Marimo page).
    """
    html, n = _DISPLAY_THEME.subn(lambda m: f"{m.group(1)}{theme}{m.group(2)}", html, count=1)
    if not n:
        return html  # not a Marimo export — nothing to theme
    scheme = _COLOR_SCHEME.get(theme, "light dark")
    html = re.sub(
        r"(<head[^>]*>)",
        lambda m: f'{m.group(1)}\n    <meta name="color-scheme" content="{scheme}" />',
        html,
        count=1,
    )
    if theme == "system":
        html = re.sub(r"(<body[^>]*>)", lambda m: f"{m.group(1)}\n    {_FLASH_GUARD}", html, count=1)
    return html


# Marimo's "Static marimo notebook — Run or Edit" banner is rendered *client-side* by
# its bundle (it's nowhere in the exported HTML — only ``data-testid`` survives at
# runtime), so we can't rewrite it as markup. We hide it via this rule on that stable
# testid; if a future Marimo drops the testid the rule simply no-ops (its banner returns,
# ours still shows) — no hard dependency on its internals.
_HIDE_MARIMO_BANNER = '[data-testid="static-notebook-banner"]{display:none!important}'

# Our nav is a ``position:fixed`` overlay, deliberately *out of normal flow*: Marimo
# mounts its app as a full-viewport ``absolute`` layer, so an in-flow bar would both be
# painted over by it and add its own height below it (a second scrollbar). Fixed + a top
# z-index floats above that layer and touches nothing in Marimo's DOM. Pinned top-left to
# clear Marimo's top-right actions (``…``) menu. ``Canvas``/``CanvasText`` are the UA's
# theme-aware system colors (the export declares ``color-scheme``, so they track the
# device theme); a blurred translucent backdrop keeps it legible over content.
_BANNER_STYLE = (
    "position:fixed;top:.5rem;left:.5rem;z-index:2147483647;"
    "display:flex;gap:.75rem;align-items:center;"
    "padding:.3rem .65rem;font-size:.8125rem;line-height:1.4;"
    "font-family:system-ui,sans-serif;border-radius:.375rem;"
    "background:color-mix(in srgb, Canvas 80%, transparent);"
    "border:1px solid color-mix(in srgb, CanvasText 18%, transparent);"
    "-webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px);"
)
_BANNER_LINK = "color:inherit;text-decoration:underline"


def set_banner(html: str, *, index_url: str | None = None, source_url: str | None = None) -> str:
    """Give a published report a floating nav — back to the index, out to the source.

    Marimo's static export shows a "Run or Edit" banner whose only action is a download
    popup; on a published site a back-link to the index and a link to the source notebook
    are more useful. So we hide Marimo's banner (a CSS rule keyed on its ``data-testid``)
    and inject our own — a small ``position:fixed`` overlay (``← Index`` · ``Source``).
    It's fixed rather than in-flow because Marimo renders its app as a full-viewport
    ``absolute`` layer that would otherwise paint over an in-flow bar and leave a second
    scrollbar below it. Either link is omitted when its URL is ``None``; a no-op if
    neither is given.
    """
    if index_url is None and source_url is None:
        return html

    def link(href: str, label: str) -> str:
        return f'<a href="{href}" style="{_BANNER_LINK}">{label}</a>'

    links = [link(url, label) for url, label in ((index_url, "&larr; Index"), (source_url, "Source")) if url]
    bar = f'<nav data-mini-banner style="{_BANNER_STYLE}">{"".join(links)}</nav>'

    html = re.sub(
        r"(</head>)",
        lambda m: (
            f"    <style>{_HIDE_MARIMO_BANNER}\n    @media print{{[data-mini-banner]{{display:none}}}}</style>\n{m.group(1)}"
        ),
        html,
        count=1,
    )
    return re.sub(r"(<body[^>]*>)", lambda m: f"{m.group(1)}\n    {bar}", html, count=1)

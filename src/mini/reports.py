"""
Report bundles: produce a report's assets as relative URLs, then repoint them.

A report is a **bundle** — one Marimo HTML document plus its heavy assets (figures,
data blobs). The two halves of the bundle protocol both live here:

**Produce.** A :class:`Publisher` writes each asset out as a content-addressed file
beside the exported HTML and hands back a *relative* URL like
``_assets/<sha>/<name>.png``. The path carries the content hash (so identical bytes
are written once and shared) *and* a readable leaf (so a browser saving the asset
suggests a sensible filename — the URL's last segment, since the bucket sets no
``Content-Disposition``). ``themed`` figures externalize through a publisher when one
is set; :meth:`Publisher.asset_url` is the general verb for any blob.

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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

__all__ = [
    'Publisher',
    'report_bundle',
    'use_publisher',
    'current_publisher',
    'relative_urls',
    'stray_links',
    'rewrite_links',
    'insert_base',
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Produce: writing a report's assets as files referenced by a relative URL
# ---------------------------------------------------------------------------


def _safe_leaf(name: str) -> str:
    """A filesystem/URL-safe leaf filename from *name* (its readable download name)."""
    leaf = re.sub(r'[^A-Za-z0-9._-]', '-', PurePosixPath(name).name)
    return leaf or 'asset'


@dataclass(frozen=True)
class Publisher:
    """Writes a report's heavy assets out as content-addressed files beside the
    exported HTML, referenced by a **relative** URL.

    Each blob is written under ``asset_dir`` (conventionally ``…/__marimo__/_assets``)
    at ``<sha256>/<name>``: the SHA directory is the content address (identical bytes
    land in one place, written once), and the readable *name* is the leaf — so a
    browser "Save as" suggests that name (it derives the filename from the URL's last
    segment, the bucket setting no ``Content-Disposition``). The reference is
    ``<link>/<sha>/<name>``; because it's relative, the same HTML resolves to the local
    files when opened off disk and to the HF bucket when published (a single
    ``<base href>`` is inserted at build time — see ``scripts/build_site.py``).
    """

    asset_dir: Path
    link: str = '_assets'

    def asset_url(self, data: bytes | Path, *, name: str) -> str:
        """Write *data* (bytes or a file) as ``<sha>/<name>`` and return its relative URL.

        Content-addressed by the SHA directory and write-once, so re-running a report
        reuses identical bytes rather than piling up copies. *name* is the readable
        download filename (carry the extension — it sets the served media type).
        """
        blob = bytes(data) if isinstance(data, (bytes, bytearray)) else Path(data).read_bytes()
        leaf = _safe_leaf(name)
        rel = f'{hashlib.sha256(blob).hexdigest()}/{leaf}'
        dest = self.asset_dir / rel
        if not dest.exists():  # content-addressed: identical bytes + name already written
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f'{leaf}.tmp')
            tmp.write_bytes(blob)
            tmp.replace(dest)  # atomic, so a concurrent reader never sees a partial file
        return f'{self.link}/{rel}'


def report_bundle(notebook_file: str | Path, *, link: str = '_assets') -> Publisher:
    """A :class:`Publisher` writing assets beside a report's exported HTML.

    Marimo exports ``docs/<…>/report.py`` to ``docs/<…>/__marimo__/report.html``; this
    points assets at ``docs/<…>/__marimo__/<link>/`` so the relative ``<link>/…`` URL
    resolves next to that HTML. Call it from the report's setup cell with ``__file__``::

        use_publisher(report_bundle(__file__))
    """
    out_dir = Path(notebook_file).resolve().parent / '__marimo__'
    return Publisher(asset_dir=out_dir / link, link=link)


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
_ANCHORED = re.compile(r'(?:[a-z][a-z0-9+.\-]*:|//|#)', re.IGNORECASE)


def relative_urls(html: str) -> list[str]:
    """Every relative ``src``/``href`` URL in *html* (escaped-in-JSON or not, in order)."""
    return [u for u in _URL_ATTR.findall(html) if u and not _ANCHORED.match(u)]


def stray_links(html: str, *, link: str = '_assets') -> list[str]:
    """Relative URLs that are *not* store assets — the ones a ``<base>`` would break.

    These are author-written nav/source links (``./experiment.py``) that should be
    absolute. Returned sorted and de-duplicated so a build can resolve or warn on them.
    """
    prefix = f'{link}/'
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
            html = html.replace(f'{q}{token}{q}', f'{q}{target}{q}')  # plain
            html = html.replace(f'\\{q}{token}\\{q}', f'\\{q}{target}\\{q}')  # escaped-in-JSON
    return html


def insert_base(html: str, href: str) -> str:
    """Insert a single ``<base href>`` as the first thing in ``<head>``.

    Placed before any resource reference so it governs all of them. Idempotent enough
    for a build step: it rewrites the first ``<head>`` only.
    """
    return re.sub(r'(<head[^>]*>)', lambda m: f'{m.group(1)}\n    <base href="{href}" />', html, count=1)

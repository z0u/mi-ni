"""
Publishing exported reports: repoint their relative asset URLs at the bucket.

A report (a Marimo HTML export) references its heavy assets — figures, data blobs —
by a *relative* URL like ``_assets/<sha>.png`` (see :class:`mini.vis.Publisher`). That
same HTML is consumed two ways:

- **opened locally**, the relative URL resolves to the co-located ``_assets/`` files;
- **served from Pages**, we want it to resolve to the assets we uploaded to the HF
  bucket instead.

The bridge is a single ``<base href>`` in the ``<head>``: it sets the document base
that *every* relative URL resolves against, so one inserted tag repoints the whole
report's assets at the bucket — no per-URL rewriting, and it works for the data URIs
buried in Marimo's session JSON (they resolve at render time) and for a relative
``fetch()`` in an interactive report alike.

The catch is that ``<base>`` is document-global, so an author-written relative *link*
(a markdown ``[src](./experiment.py)``) would be repointed too — and 404 against the
bucket. :func:`stray_links` finds those at build time so they can be made absolute;
the convention is *the only relative URLs in a report are store assets*.
"""

from __future__ import annotations

import re

__all__ = ['relative_urls', 'stray_links', 'insert_base']

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
    absolute. Returned sorted and de-duplicated so a build can warn about them.
    """
    prefix = f'{link}/'
    return sorted({u for u in relative_urls(html) if not u.startswith(prefix)})


def insert_base(html: str, href: str) -> str:
    """Insert a single ``<base href>`` as the first thing in ``<head>``.

    Placed before any resource reference so it governs all of them. Idempotent enough
    for a build step: it rewrites the first ``<head>`` only.
    """
    return re.sub(r'(<head[^>]*>)', lambda m: f'{m.group(1)}\n    <base href="{href}" />', html, count=1)

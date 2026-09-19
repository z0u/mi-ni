"""Print an exported report bundle to PDF, offline, with a headless browser.

A bundle (``mini.lit``) is a self-styled static page with its figures beside it, so :func:`served_bundle` copies it to a throwaway serve root and serves that on a loopback port. What the page pulls from elsewhere (KaTeX and the fonts, from CDNs) is fetched by Python and cached (:func:`route_remote`): a browser in a proxied sandbox cannot reach a CDN, and a print that could would still render the math of the day; from the cache, a print is offline and repeatable. :func:`print_bundle` then drives Chromium through Playwright and prints the page through the same engine as Chrome's print dialog, so the ``@page`` size and ``@media print`` rules in ``docs/report.css`` (paper sized for a reMarkable 2, one section per page) are honoured.

This runs at export (``scripts/export_reports.py``), the half of publishing that holds the bundle on disk: the PDF lands beside ``index.html``, rides the bundle sync, and is pinned by the same ``publish.lock`` entry as the page. The site build only links it. Chromium stamps a creation date and a random document ID into every PDF, which would make each re-export of an unchanged report a new publish-tier commit, so :func:`normalize_pdf` strips both after printing; two prints of one bundle are then byte-equal.

The stylesheet's page is a fixed size, and a page has one height for the whole document: tall enough for the longest section, every shorter section trails a sheet of white; short enough for a typical one, the long ones break mid-table. The print takes the height out of the equation. It grows the page (from twice the stylesheet's height, doubling toward :data:`MAX_PAGE_MM`) until the document has one page per section, then clips each page to the ink on it (:func:`ink_extents`, a low-resolution render of each page), so a page is as long as its section and no longer. The stylesheet keeps its own height for the browser's print dialog, which cannot clip.

Playwright is a dev dependency and Chromium is found via ``PLAYWRIGHT_CHROMIUM``, the cloud sandbox's ``/opt/pw-browsers/chromium``, or Playwright's own resolution. When none of those works, :func:`print_bundle` says so and returns ``None`` rather than failing the export: the PDF is a convenience beside the page, never a condition of it.
"""

from __future__ import annotations

import http.server
import logging
import os
import hashlib
import re
import shutil
import socketserver
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = [
    "served_bundle",
    "print_bundle",
    "print_page",
    "route_remote",
    "ink_extents",
    "normalize_pdf",
    "chromium_path",
    "MAX_PAGE_MM",
]

log = logging.getLogger(__name__)

# The tallest page a PDF reader is sure to open (Acrobat's 200-inch limit); the print
# grows a page toward it until every section fits on one.
MAX_PAGE_MM = 5080

_MM_PER = {"mm": 1.0, "cm": 10.0, "in": 25.4, "pt": 25.4 / 72, "px": 25.4 / 96}

# The ``size`` of the last ``@page`` rule that sets one, wherever it sits (nested in
# ``@media print``, say); the cascade's last-wins is what applies. Stylesheets from
# another origin cannot be read and are skipped.
_PAGE_SIZE_JS = """() => {
  const walk = rules => [...rules].flatMap(r =>
    r instanceof CSSPageRule ? [r.style.getPropertyValue('size')] : r.cssRules ? walk(r.cssRules) : []);
  const sizes = [...document.styleSheets].flatMap(s => { try { return walk(s.cssRules) } catch (e) { return [] } });
  return sizes.filter(Boolean).at(-1) ?? null;
}"""

# How many elements start a fresh page, under print media (emulated by the caller).
_PAGE_BREAKS_JS = """() =>
  [...document.querySelectorAll('body *')].filter(e => getComputedStyle(e).breakBefore === 'page').length"""

# A <style> last in the document, so its @page outranks the stylesheet's; re-run with a
# new size, it replaces its own text rather than stacking.
_SET_PAGE_SIZE_JS = """size => {
  let s = document.getElementById('mini-print-page');
  if (!s) { s = document.createElement('style'); s.id = 'mini-print-page'; document.documentElement.appendChild(s); }
  s.textContent = `@page { size: ${size}; }`;
}"""

INSTALL_HINT = (
    "uv run playwright install chromium && uv run playwright install-deps chromium  (one download, then cached)"
)


def chromium_path() -> str | None:
    """A Chromium binary to launch, or ``None`` to let Playwright resolve its own."""
    exe = os.environ.get("PLAYWRIGHT_CHROMIUM", "/opt/pw-browsers/chromium")
    return exe if Path(exe).exists() else None


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """A page and its figures are a few dozen requests; an export log has no use for them."""

    def log_message(self, format: str, *args: Any) -> None:
        pass


def _build_serve_root(bundle: Path, root: Path, *, html: str | None = None) -> None:
    """Assemble a serve root: the bundle's ``_assets/`` plus its page (or *html* in its place)."""
    # Absolute, so the copies below resolve regardless of the bundle's cwd.
    index = (bundle / "index.html" if bundle.is_dir() else bundle).resolve()
    assets = index.parent / "_assets"

    # Copies, not symlinks: a write into the serve root (like index.html below) must
    # never reach through a link into the bundle.
    if assets.is_dir():
        shutil.copytree(assets, root / "_assets")

    (root / "index.html").write_text(html if html is not None else index.read_text("utf-8"), "utf-8")


@contextmanager
def served_bundle(bundle: Path, *, html: str | None = None) -> Iterator[str]:
    """Serve *bundle* (a dir with ``index.html`` + ``_assets/``, or the ``index.html``) offline; yields its URL.

    The serve root is a throwaway copy beside the bundle (``.render-<name>/``, gitignored under ``.mini``), removed on exit. *html* replaces the bundle's page for the duration, for a caller that has rewritten it (resolved links, say) without wanting that in the bundle itself.
    """
    root = bundle.parent / (".render-" + (bundle.name or "root"))
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    handler = lambda *a, **k: _QuietHandler(*a, directory=str(root), **k)  # noqa: E731
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    try:
        _build_serve_root(bundle, root, html=html)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
    finally:
        httpd.shutdown()
        httpd.server_close()
        shutil.rmtree(root, ignore_errors=True)


def cache_dir() -> Path:
    """Where fetched remote resources are kept: ``$XDG_CACHE_HOME/mini/remote`` (``~/.cache/...``)."""
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "mini" / "remote"


def _fetch(url: str, headers: dict[str, str]) -> tuple[str, bytes]:
    """GET *url* with Python's TLS (which honours the proxy's CA); returns (content type, body)."""
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return r.headers.get("content-type", "application/octet-stream"), r.read()


Fetch = Callable[[str, dict[str, str]], tuple[str, bytes]]


def route_remote(page: Any, *, cache: Path | None = None, fetch: Fetch = _fetch) -> None:
    """Serve *page*'s ``https://`` requests from a cache that Python fills, rather than the browser's own network.

    The page's stylesheet links KaTeX and the fonts from CDNs. A headless Chromium in a proxied sandbox rejects the proxy's certificate and gets neither, so the math printed as its source; Python trusts the certificate (``SSL_CERT_FILE``), so it fetches instead, once per URL, into :func:`cache_dir`. The request's own headers go along, since Google Fonts picks the font format by user agent. A fetch that fails is aborted, with one warning per host, and the print goes on without it: the PDF is a convenience beside the page.
    """
    root = cache if cache is not None else cache_dir()
    warned: set[str] = set()

    def handle(route: Any) -> None:
        req = route.request
        if req.method != "GET":
            route.continue_()
            return
        key = hashlib.sha256(req.url.encode()).hexdigest()
        body, kind = root / key, root / (key + ".type")
        if not body.exists():
            try:
                content_type, data = fetch(req.url, {k: v for k, v in req.headers.items() if k.lower() != "host"})
            except (urllib.error.URLError, OSError, ValueError) as e:
                host = req.url.split("/")[2]
                if host not in warned:
                    warned.add(host)
                    log.warning("print: %s unreachable (%s); printing without it", host, e)
                route.abort()
                return
            root.mkdir(parents=True, exist_ok=True)
            body.write_bytes(data)
            kind.write_text(content_type)
        route.fulfill(status=200, content_type=kind.read_text(), body=body.read_bytes())

    page.route(re.compile(r"^https://"), handle)


def print_page(page: Any, out: Path, *, settle: float = 3.0, fit: bool = True) -> Path:
    """Print an already-loaded Playwright *page* to *out* and normalize the file.

    Same engine as Chrome's print dialog: the size and margins come from the stylesheet's ``@page`` rule, and there is no header or footer. Light scheme, since the reader is paper or e-ink. Figures the site build deferred have no pixels until scrolled to (the site's ``beforeprint`` handler does the same, but headless printing may not fire it), so they are made eager and given *settle* seconds to arrive; an export bundle carries no such deferral and pays only the wait.

    With *fit* (the default), a page that the stylesheet sizes in absolute units is printed tall enough that no section breaks across pages, and each page is then clipped to its content (see the module docstring). A page with no ``@page`` size, or a named one (``A4``), prints as the stylesheet says.
    """
    page.emulate_media(color_scheme="light")
    page.evaluate("document.querySelectorAll('img[loading=lazy]').forEach(i => i.loading = 'eager')")
    page.wait_for_timeout(settle * 1000)
    size = _page_size_mm(page) if fit else None
    if size is None:
        page.pdf(path=str(out), prefer_css_page_size=True, print_background=True)
        normalize_pdf(out)
        return out
    _print_fitting_sections(page, out, size)
    normalize_pdf(out, extents=ink_extents(out))
    return out


def _page_size_mm(page: Any) -> tuple[float, float] | None:
    """The stylesheet's ``@page`` size as (width, height) in mm, or ``None`` when it is absent or a named size."""
    size = page.evaluate(_PAGE_SIZE_JS)
    parts = (size or "").split()
    if len(parts) == 1:  # a square page, "100mm"
        parts *= 2
    if len(parts) != 2:
        return None
    dims = []
    for part in parts:
        m = re.fullmatch(r"([0-9.]+)(mm|cm|in|pt|px)", part)
        if m is None:  # a named size (A4), a keyword (landscape), or a value the units map lacks
            return None
        dims.append(float(m.group(1)) * _MM_PER[m.group(2)])
    return dims[0], dims[1]


def _print_fitting_sections(page: Any, out: Path, size: tuple[float, float]) -> None:
    """Print *page* on a sheet tall enough that each section is one page.

    A section is what a ``break-before: page`` starts (the report stylesheet puts one on each H2), so a document with *n* breaks fits when it prints to *n* + 1 pages. Starting at twice the stylesheet's height and doubling from there, capped at :data:`MAX_PAGE_MM`; the clip afterwards means the slack costs nothing.
    """
    import pikepdf

    width, height = size
    page.emulate_media(media="print", color_scheme="light")
    sections = 1 + page.evaluate(_PAGE_BREAKS_JS)
    height = min(2 * height, MAX_PAGE_MM)
    while True:
        page.evaluate(_SET_PAGE_SIZE_JS, f"{width:g}mm {height:g}mm")
        page.pdf(path=str(out), prefer_css_page_size=True, print_background=True)
        with pikepdf.open(out) as pdf:
            pages = len(pdf.pages)
        if pages <= sections or height >= MAX_PAGE_MM:
            break
        height = min(2 * height, MAX_PAGE_MM)
    if pages > sections:
        log.warning("report PDF: %d section(s) print to %d pages at the tallest page (%d mm)", sections, pages, height)


def ink_extents(path: Path, *, dpi: float = 18) -> list[tuple[float, float] | None]:
    """Per page, the top and bottom of the ink measured from the page's top edge, in points; ``None`` for a blank page.

    A low-resolution render of each page (*dpi*), read for the first and last row that is not paper-white: anything drawn counts, an admonition's tint as much as a glyph. Coarse on purpose, since the clip below adds a margin far wider than a pixel.
    """
    import numpy as np
    import pypdfium2 as pdfium

    scale = dpi / 72
    out: list[tuple[float, float] | None] = []
    doc = pdfium.PdfDocument(str(path))
    try:
        for pg in doc:
            pixels = pg.render(scale=scale).to_numpy()
            rows = np.flatnonzero(np.any(pixels[..., :3] < 250, axis=(1, 2)))
            out.append((float(rows[0]) / scale, float(rows[-1] + 1) / scale) if rows.size else None)
    finally:
        doc.close()
    return out


def normalize_pdf(path: Path, *, extents: list[tuple[float, float] | None] | None = None) -> None:
    """Strip what varies between two prints of one page, so the bytes are a function of the content; clip pages to *extents*.

    Chromium stamps ``CreationDate``/``ModDate`` (now) and a document ID (random) into every PDF. A re-export of an unchanged report would then upload a different file and mint a publish-tier commit for nothing, where today an identical bundle mints none. Dates go; the ID is re-derived from the content (qpdf's deterministic ID).

    *extents* (from :func:`ink_extents`) clips each page's box to its ink, keeping the top edge: the white below the last ink is made the same as the white above the first, which is the top margin plus the heading's leading, so the two ends of a page match. A blank page is left as it is.

    In-page links (a footnote and its backlink, a heading) print as named destinations in the document's ``/Dests`` dictionary; each link annotation is given its destination outright (:func:`_inline_dests`), so a viewer that resolves only direct destinations, as the simpler e-ink ones do, follows them too.
    """
    import pikepdf

    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        for key in ("/CreationDate", "/ModDate"):
            if key in pdf.docinfo:
                del pdf.docinfo[key]
        _inline_dests(pdf)
        for page, extent in zip(pdf.pages, extents or [], strict=extents is not None):
            if extent is None:
                continue
            top, bottom = extent
            x0, y0, x1, y1 = (float(v) for v in page.MediaBox)  # PDF y runs up: y1 is the top edge
            page.MediaBox = page.CropBox = [x0, max(y0, y1 - bottom - top), x1, y1]
        pdf.save(path, deterministic_id=True)


def _inline_dests(pdf: Any) -> None:
    """Replace each link annotation's named destination with the array the name resolves to (the ``/Dests`` dictionary is left for viewers that read it)."""
    import pikepdf

    dests = pdf.Root.get("/Dests")
    if dests is None:
        return
    for page in pdf.pages:
        for annot in page.get("/Annots") or []:
            name = annot.get("/Dest")
            if name is None or isinstance(name, pikepdf.Array):
                continue  # no destination, or one given outright already
            key = str(name)  # a Name (``/fn:a``) or a String (``fn:a``); the dictionary keys by Name
            target = dests.get(key if key.startswith("/") else "/" + key)
            if isinstance(target, pikepdf.Array):
                annot.Dest = target


def print_bundle(
    bundle: Path, out: Path, *, html: str | None = None, timeout: float = 8.0, settle: float = 3.0
) -> Path | None:
    """Print an export bundle to *out*; ``None`` (with a log line) when no browser is available.

    *html*, if given, is printed in place of the bundle's page (see :func:`served_bundle`). Waits up to *timeout* seconds for the content (``main.lit``) to appear, then *settle* seconds for figures and fonts. A missing Playwright or Chromium is reported with the install commands and never raises: a publish must not fail on the PDF.
    """
    try:
        from playwright.sync_api import Error as PlaywrightError, sync_playwright
    except ImportError:
        log.warning("report PDF skipped: playwright is not installed (uv sync --all-groups)")
        return None
    with served_bundle(bundle, html=html) as url, sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=chromium_path())
        except PlaywrightError as e:
            log.warning(
                "report PDF skipped: no Chromium to print with (%s). Install once: %s",
                str(e).splitlines()[0],
                INSTALL_HINT,
            )
            return None
        try:
            # A bare headless Chromium in a locale-less container reports no
            # navigator.language; pin one so nothing on the page has to guess.
            page = browser.new_page(viewport={"width": 1100, "height": 1400}, locale="en-US")
            route_remote(page)
            page.goto(url)
            page.locator("main.lit").first.wait_for(timeout=timeout * 1000)
            if page.locator(".arithmatex").count() and not page.locator(".katex").count():
                log.warning("print: the page has math but KaTeX did not render it; the PDF shows the source")
            return print_page(page, out, settle=settle)
        finally:
            browser.close()

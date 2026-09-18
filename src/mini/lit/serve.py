"""
Watch a script, re-weave it on save, and serve it with a live reload.

The editor stays whatever you write in; the browser is the viewer. The page goes to ``.mini/lit-live/<key>/``, apart from where ``render`` writes, so the two can run at once (they share only the memo cache, which is content-keyed). One process holds the :class:`~mini.lit.document.Runner` (so unchanged cells are not re-run and ``@memo`` hits are in memory), polls the document and the ``.py`` files beside it for changes, rewrites ``index.html`` when something moved, and answers a long-poll from the page so the browser reloads the moment a build lands. A sibling ``.py`` edit (a helper module beside the document) drops that module from ``sys.modules`` and resets the runner, since any cell may have imported it.

The server listens before the first build, and a build shows its work: when a cell is still running after a moment (a download, a fit), the page is replaced with the document as it stands — everything above it, a running note, and the prose below with pending marks — so a slow cell never hides the rest of the report. The real page replaces it when the build lands.
"""

from __future__ import annotations

import html
import sys
import threading
import time
from collections.abc import Callable
from functools import partial as bind
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mini.lit.document import Woven
from mini.lit.render import Rendered, compose, output_dir, render
from mini.lit.render import _write as write_file

__all__ = ["serve"]

# How long a cell may run before the page shows it as running. Short enough that a
# download is visible at once, long enough that the ordinary run of quick cells does not
# reload the page once per cell.
PARTIAL_AFTER = 0.3

_RELOAD = """
<script>
(async () => {
  let v = %d;
  for (;;) {
    try {
      const r = await fetch(`/__version?after=${v}`, {cache: "no-store"});
      const n = Number(await r.text());
      if (n !== v) location.reload();
    } catch (e) { await new Promise(r => setTimeout(r, 1000)); }
  }
})();
</script>
"""


class _Site:
    """The served directory and its version, which the page long-polls; every write goes through :meth:`publish`."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.version = 0
        self.changed = threading.Condition()
        self.lock = threading.Lock()

    def publish(self, html_for: Callable[[str], str], markdown: str | None = None) -> None:
        """Write the page *html_for* builds around the reload script for the next version, then bump."""
        with self.lock:
            with self.changed:
                page = html_for(_RELOAD % (self.version + 1))
                write_file(self.out / "index.html", page)
                if markdown is not None:
                    write_file(self.out / "index.md", markdown)
                self.version += 1
                self.changed.notify_all()

    def wait_past(self, v: int, timeout: float) -> int:
        with self.changed:
            self.changed.wait_for(lambda: self.version != v, timeout=timeout)
            return self.version


class _Handler(SimpleHTTPRequestHandler):
    site: _Site

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/__version"):
            after = int(self.path.rpartition("=")[2] or 0)
            body = str(self.site.wait_past(after, timeout=25)).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", ""):
            self.path = "/index.html"
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:  # quiet
        pass


class _Builder:
    """One build at a time: weave, publishing a partial page when a cell is slow, then the final page."""

    def __init__(self, doc: Path, site: _Site) -> None:
        self.doc = doc
        self.site = site
        self.runner = None
        self._timer: threading.Timer | None = None
        self._live = False  # whether a pending partial may still publish

    def _partial(self, woven: Woven) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(PARTIAL_AFTER, self._publish_partial, [woven])
        self._timer.daemon = True
        self._timer.start()

    def _publish_partial(self, woven: Woven) -> None:
        with self.site.lock:
            if not self._live:  # the build finished first
                return
        cell = woven.running
        self.site.publish(lambda reload: compose(woven, extra_body=reload)[0])
        print(
            f"{self.doc.name}: running the cell at line {cell.line if cell else '?'}, page shows the rest", flush=True
        )

    def build(self) -> None:
        self._live = True
        try:
            r = render(
                self.doc, out_dir=self.site.out, live=True, runner=self.runner, write=False, partial=self._partial
            )
            self.runner = r.runner
        finally:
            with self.site.lock:
                self._live = False
            if self._timer is not None:
                self._timer.cancel()
        self.site.publish(lambda reload: compose(r.woven, extra_body=reload)[0], markdown=r.woven.markdown)
        _report(r)


def _mtimes(doc: Path) -> dict[Path, float]:
    files = [doc, *doc.parent.glob("*.py")]
    return {f: f.stat().st_mtime for f in files if f.exists()}


def _watch(doc: Path, builder: _Builder, poll: float) -> None:
    """Build once, then poll the document and its sibling modules and rebuild on a change (resetting the runner if a module moved)."""
    seen = _mtimes(doc)
    builder.build()
    while True:
        time.sleep(poll)
        now = _mtimes(doc)
        if now == seen:
            continue
        for f in {k for k in now.keys() | seen.keys() if now.get(k) != seen.get(k)}:
            if f.suffix == ".py" and f != doc:  # a sibling module (the document itself is the runner's business)
                for name, mod in list(sys.modules.items()):
                    if getattr(mod, "__file__", None) == str(f):
                        del sys.modules[name]
                if builder.runner is not None:
                    builder.runner.reset()
        seen = now
        try:
            builder.build()
        except Exception as e:  # keep serving the last good page
            print(f"build failed: {e}", file=sys.stderr, flush=True)


def serve(
    doc: Path | str, *, out_dir: Path | None = None, port: int = 8765, host: str = "127.0.0.1", poll: float = 0.25
) -> None:
    doc = Path(doc).resolve()
    out = (out_dir or output_dir(doc, live=True)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    site = _Site(out)
    placeholder = f"<!doctype html><title>{html.escape(doc.name)}</title><p>Rendering {html.escape(doc.name)}…</p>"
    site.publish(lambda reload: placeholder + reload)
    threading.Thread(target=_watch, args=(doc, _Builder(doc, site), poll), daemon=True).start()
    handler = type("Handler", (_Handler,), {"site": site})
    server = ThreadingHTTPServer((host, port), bind(handler, directory=str(out)))
    print(f"Serving {doc.name} at http://localhost:{port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


def _report(r: Rendered) -> None:
    w = r.woven
    note = f", {len(w.errors)} error(s)" if w.errors else (", stopped early" if w.stopped else "")
    print(
        f"{r.doc.path.name}: {w.cells_run} cell(s) run, woven in {w.seconds * 1e3:.0f} ms, page in {r.seconds * 1e3:.0f} ms{note}",
        flush=True,
    )

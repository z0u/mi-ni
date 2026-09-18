"""
Watch a document, re-weave it on save, and serve it with a live reload.

The editor stays whatever you write in; the browser is the viewer. One process holds the :class:`~mini.lit.document.Runner` (so unchanged cells are not re-run and ``@memo`` hits are in memory), polls the document and the ``.py`` files beside it for changes, rewrites ``index.html`` when something moved, and answers a long-poll from the page so the browser reloads the moment a build lands. A sibling ``.py`` edit drops that module from ``sys.modules`` and resets the runner, since any cell may have imported it.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mini.lit.render import Rendered, render

__all__ = ["serve"]

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


class _State:
    def __init__(self) -> None:
        self.version = 0
        self.changed = threading.Condition()

    def bump(self) -> None:
        with self.changed:
            self.version += 1
            self.changed.notify_all()

    def wait_past(self, v: int, timeout: float) -> int:
        with self.changed:
            self.changed.wait_for(lambda: self.version != v, timeout=timeout)
            return self.version


class _Handler(SimpleHTTPRequestHandler):
    state: _State

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/__version"):
            after = int(self.path.rpartition("=")[2] or 0)
            v = self.state.wait_past(after, timeout=25)
            body = str(v).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
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


def _mtimes(doc: Path) -> dict[Path, float]:
    files = [doc, *doc.parent.glob("*.py")]
    return {f: f.stat().st_mtime for f in files if f.exists()}


def serve(
    doc: Path | str, *, out_dir: Path | None = None, port: int = 8765, host: str = "127.0.0.1", poll: float = 0.25
) -> None:
    doc = Path(doc).resolve()
    state = _State()
    rendered: Rendered = render(doc, out_dir=out_dir, live=True, extra_body=_RELOAD % 0)
    out = rendered.out_dir
    _report(rendered)

    def rebuild() -> None:
        nonlocal rendered
        rendered = render(doc, out_dir=out, live=True, runner=rendered.runner, extra_body=_RELOAD % (state.version + 1))
        _report(rendered)
        state.bump()

    threading.Thread(target=_watch, args=(doc, poll, rendered.runner.reset, rebuild), daemon=True).start()
    handler = type("Handler", (_Handler,), {"state": state})
    server = ThreadingHTTPServer((host, port), partial(handler, directory=str(out)))
    print(f"Serving {doc.name} at http://localhost:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


def _watch(doc: Path, poll: float, reset: Callable[[], None], rebuild: Callable[[], None]) -> None:
    """Poll the document and its sibling modules; on a change, rebuild (after resetting the runner if a module moved)."""
    seen = _mtimes(doc)
    while True:
        time.sleep(poll)
        now = _mtimes(doc)
        if now == seen:
            continue
        for f in {k for k in now.keys() | seen.keys() if now.get(k) != seen.get(k)}:
            if f.suffix == ".py":
                for name, mod in list(sys.modules.items()):
                    if getattr(mod, "__file__", None) == str(f):
                        del sys.modules[name]
                reset()
        seen = now
        try:
            rebuild()
        except Exception as e:  # keep serving the last good page
            print(f"build failed: {e}", file=sys.stderr)


def _report(r: Rendered) -> None:
    w = r.woven
    note = f", {len(w.errors)} error(s)" if w.errors else (", stopped early" if w.stopped else "")
    print(
        f"{r.doc.path.name}: {w.cells_run} cell(s) run, woven in {w.seconds * 1e3:.0f} ms, page in {r.seconds * 1e3:.0f} ms{note}"
    )

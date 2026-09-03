#!/usr/bin/env python
"""Serve a locally-assembled ``_site/`` with GitHub Pages semantics.

``npx serve`` (serve-handler) decides file-vs-directory from ``path.extname()``, a string check — so it mistakes our dotted report dirs (``m1/ex-2.9.1``) for files, skips its ``index.html`` lookup, and renders a directory listing. Working around it with redirects means emitting 301s, which browsers cache permanently and replay as loops long after the config is fixed.

So we stat the filesystem instead, matching GitHub Pages: a directory with no ``index.html`` is a 404; one that has it redirects to the trailing-slash form (so the page's relative ``_assets/`` resolve against ``…/ex-2.1.1/``, not its parent) and then serves the index. The redirect is a 302, not GitHub's 301 — a preview server churns, and a cached permanent redirect is the exact footgun that made us drop ``npx serve``.
"""

import sys
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        fs = Path(self.translate_path(self.path))
        if fs.is_dir():
            url_path = self.path.split("?", 1)[0].split("#", 1)[0]
            if not (fs / "index.html").is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "No index.html in directory")
                return None
            if not url_path.endswith("/"):
                self.send_response(HTTPStatus.FOUND)  # 302: transient, never cached
                self.send_header("Location", f"{url_path}/")
                self.end_headers()
                return None
            self.path = f"{url_path}index.html"
        return super().send_head()


def main() -> int:
    root = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    # Loopback only: SimpleHTTPRequestHandler follows symlinks, and a preview is for
    # this machine — binding every interface would offer the tree (and whatever a stray
    # link resolves to) to the whole network. Pass a host explicitly to opt out.
    host = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1"
    server = ThreadingHTTPServer((host, port), partial(Handler, directory=root))
    print(f"Serving {root} at http://localhost:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

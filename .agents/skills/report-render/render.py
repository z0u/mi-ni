#!/usr/bin/env python
"""Render a marimo report export in a headless browser — offline.

A `marimo export html` bundle loads its frontend runtime (~200 JS/CSS/font URLs)
from the jsDelivr CDN, so it won't render in a network-restricted sandbox. But the
*same* pinned `dist/` ships inside the marimo pip package under `_static/`. This
repoints the bundle's CDN refs at those local assets, serves the result, and drives
the pre-installed Chromium — letting you screenshot or assert on a report's real
rendered DOM (figures, layout, the show-code toggle) without any network.

Run it through the project env (so the local _static/ assets match the pinned
marimo that produced the bundle), adding Playwright just for this call:

    uv run --with playwright python .claude/skills/report-render/render.py \
        .mini/exports/gpt-sweep -o /tmp/report.png

Pass a bundle dir (containing index.html + _assets/) or an index.html directly.
`--suffix '?show-code=true'` appends to the URL; `--wait-text STR` blocks until STR
appears (or times out). `--selector CSS` shoots just the matching element(s) instead
of the full page — e.g. `--selector '.output svg'` for one figure, numbering the
output when several match. See SKILL.md for driving the DOM instead of screenshotting.
"""

import argparse
import http.server
import os
import re
import shutil
import socketserver
import threading
from pathlib import Path

import marimo

# The whole CDN base every asset URL shares: .../@marimo-team/frontend@<version>/dist
_CDN = re.compile(r"https://cdn\.jsdelivr\.net/npm/@marimo-team/frontend@[^/\"']+/dist")


def _build_serve_root(bundle: Path, root: Path) -> None:
    """Assemble a serve root: marimo's _static assets + the bundle's CDN-rewritten HTML."""
    # Absolute, so the copies below resolve regardless of the bundle's cwd.
    index = (bundle / "index.html" if bundle.is_dir() else bundle).resolve()
    assets = index.parent / "_assets"
    static = Path(marimo.__file__).parent / "_static"

    # marimo runtime lives under assets/ (+ favicon etc.); copy it all in at root, so a
    # rewritten "/assets/index-*.js" resolves here. The report's figures live under _assets/
    # (note the leading underscore) — a different dir, so no collision. Copies, not
    # symlinks: a write into the serve root (like index.html below) must never reach
    # through a link into the marimo package or the bundle — that once corrupted marimo's
    # export template in site-packages (and, via uv's hardlinks, the uv cache), poisoning
    # every later `marimo export`. The runtime is a few tens of MB, copied into a
    # throwaway dir; self-contained beats cheap here.
    shutil.copytree(static, root, dirs_exist_ok=True)
    if assets.is_dir():
        shutil.copytree(assets, root / "_assets")

    html = _CDN.sub("", index.read_text("utf-8"))  # ".../dist/assets/x.js" -> "/assets/x.js"
    (root / "index.html").write_text(html, "utf-8")


def _out_paths(out: Path, n: int) -> list[Path]:
    """One path for a single shot, else `out` with a `-<i>` index before its suffix."""
    return [out] if n == 1 else [out.with_stem(f"{out.stem}-{i}") for i in range(n)]


def _serve(root: Path) -> tuple[socketserver.TCPServer, int]:
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(root), **k)  # noqa: E731
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path, help="export bundle dir (with index.html + _assets/) or an index.html")
    ap.add_argument("-o", "--out", type=Path, default=Path("report.png"), help="screenshot path (PNG)")
    ap.add_argument("--suffix", default="", help="appended to the URL, e.g. '?show-code=true'")
    ap.add_argument("--selector", default=None, help="CSS: shoot matching element(s), not the full page")
    ap.add_argument("--wait-text", default=None, help="block until this text appears (else fixed timeout)")
    ap.add_argument("--timeout", type=float, default=6.0, help="seconds to wait for the app to settle")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright  # ty: ignore[unresolved-import]  # runtime-only (uv run --with playwright)

    # A tmp serve root beside the bundle; throwaway copies, gitignored under .mini.
    root = args.bundle.parent / (".render-" + (args.bundle.name or "root"))
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    try:
        _build_serve_root(args.bundle, root)
        httpd, port = _serve(root)
        exe = os.environ.get("PLAYWRIGHT_CHROMIUM", "/opt/pw-browsers/chromium")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=exe if Path(exe).exists() else None)
            # marimo's frontend validates navigator.language on boot and hard-errors
            # ("Incorrect locale information provided") if the browser reports none —
            # which a bare headless Chromium in a locale-less container does. Pin one.
            page = browser.new_page(viewport={"width": 1100, "height": 1400}, locale="en-US")
            page.goto(f"http://127.0.0.1:{port}/index.html{args.suffix}")
            if args.wait_text:
                page.get_by_text(args.wait_text).first.wait_for(timeout=args.timeout * 1000)
            else:
                page.wait_for_timeout(args.timeout * 1000)
            if args.selector:
                loc = page.locator(args.selector)
                n = loc.count()
                if n == 0:
                    raise SystemExit(f"no elements match {args.selector!r}")
                outs = _out_paths(args.out, n)
                for i, out in enumerate(outs):
                    el = loc.nth(i)
                    el.scroll_into_view_if_needed()
                    el.screenshot(path=str(out))
                shot = f"{n} element(s) matching {args.selector!r} -> " + ", ".join(map(str, outs))
            else:
                page.screenshot(path=str(args.out), full_page=True)
                shot = str(args.out)
            browser.close()
        httpd.shutdown()
        print(f"rendered {args.bundle} -> {shot}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()

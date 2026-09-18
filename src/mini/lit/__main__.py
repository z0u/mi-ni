"""``python -m mini.lit``: render a literate document (``.py`` or ``.md``), or serve it with live reload."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mini.lit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="weave and write index.html (+ index.md) under .mini/lit/<key>/")
    r.add_argument("doc", type=Path)
    r.add_argument("-o", "--out", type=Path, default=None, help="output directory")
    r.add_argument("--pdf", action="store_true", help="also print the page to index.pdf with headless Chromium")
    s = sub.add_parser("serve", help="watch the document, re-render on save, and serve it with live reload")
    s.add_argument("doc", type=Path)
    s.add_argument("-o", "--out", type=Path, default=None)
    s.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        from mini.lit.serve import serve

        serve(args.doc, out_dir=args.out, port=args.port)
        return 0

    from mini.lit.render import render, to_pdf

    t0 = time.perf_counter()
    res = render(args.doc, out_dir=args.out)
    w = res.woven
    print(
        f"{res.out_dir / 'index.html'}: {w.cells_run} cell(s) run, woven in {w.seconds * 1e3:.0f} ms, page in {res.seconds * 1e3:.0f} ms, total {(time.perf_counter() - t0) * 1e3:.0f} ms"
    )
    for o in w.errors:
        print(f"error in cell at line {o.cell.line}:\n{o.error}", file=sys.stderr)
    if args.pdf:
        print(to_pdf(res.out_dir / "index.html"))
    return 1 if w.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

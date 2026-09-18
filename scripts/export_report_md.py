#!/usr/bin/env python
"""Render a report to Markdown with its outputs, for reading as text.

Usage: ``./go render <report.py>``, which is this script with the output defaulted to :func:`~mini.reports.render_path` and the mtime staleness check on. The report is woven by :func:`mini.lit.render`: prose and cell outputs in one Markdown document, figures beside it under ``<stem>.assets/``. A render is what a text-only reader (an agent, mostly) reads in place of the page.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

from mini.lit import render
from mini.reports import is_stale, render_path

_ASSET_REF_RE = re.compile(r'(?<=[("])_assets/')  # a woven figure's link: `src="_assets/…"`, `href="…"`, or `](…)`


def weave(script: Path, dst: Path, *, assets_dir: Path | None = None) -> None:
    """Write a literate script's woven Markdown to *dst*, its figures beside it under ``<stem>.assets/``.

    :func:`mini.lit.render` weaves prose and cells into one Markdown document with the figures under an ``_assets/`` dir of its own, the shape a published bundle has. A render is one file with its images in a sibling dir named for it, so the figures are moved there and the links repointed. The render happens in a temporary dir beside the output, so a failed weave leaves the last render in place.
    """
    assets_dir = assets_dir or dst.with_name(f"{dst.stem}.assets")
    rel_dir = assets_dir.name if assets_dir.parent == dst.parent else str(assets_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dst.stem}-", dir=dst.parent) as tmp:
        rendered = render(script, out_dir=Path(tmp), write=False)
        if errors := rendered.woven.errors:
            lines = "\n\n".join(f"cell at line {o.cell.line}:\n{o.error}" for o in errors)
            sys.exit(f"render of {script} failed: {len(errors)} cell(s) raised\n{lines}")
        shutil.rmtree(assets_dir, ignore_errors=True)
        if (written := Path(tmp) / "_assets").is_dir():
            shutil.move(written, assets_dir)
    dst.write_text(_ASSET_REF_RE.sub(f"{rel_dir}/", rendered.woven.markdown))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path, nargs="?", help="defaults to .mini/renders/<key>.md")
    parser.add_argument("--force", action="store_true", help="re-render even if the output looks up to date")
    parser.add_argument("--assets-dir", type=Path, help="defaults to <output-stem>.assets/ beside the output")
    args = parser.parse_args()

    dst = args.output or render_path(args.report)
    # Same mtime heuristic the bundle export uses for `--stale-only`. Rendering re-runs the
    # report, so a repeat pass over an unedited one costs time for a byte-identical file.
    if not args.force and not is_stale(args.report, dst):
        print(f"fresh  {dst} (newer than the report and its inputs — `--force` re-renders)")
        return

    weave(args.report, dst, assets_dir=args.assets_dir)
    print(f"render {args.report} -> {dst}")


if __name__ == "__main__":
    main()

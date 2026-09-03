#!/usr/bin/env python
"""Export a Marimo notebook to Markdown with its outputs, then clean it up.

This replaces the ``marimo-md-export`` console script. It reuses that package's library — which does the hard part: running both the Markdown and HTML exports, scraping rendered outputs out of the HTML session JSON, and injecting them back into the Markdown — but fixes two things in how its CLI drives that library:

**Admonitions were rewritten inside code fences.** ``marimo-md-export`` applies its ``/// type | Title`` → ``!!! type "Title"`` conversion to the whole Markdown document before collecting cells. Cells are matched to their rendered output by MD5 of the cell source, so rewriting a ``///`` block that lives *inside* a cell's Python source changes that cell's hash, and the output no longer matches. A cell whose code is hidden and whose output does not match is deleted outright, so the cell silently disappears from the export. This hits any ``mo.md(f"...")`` cell containing an admonition: interpolated cells stay code fences in the Markdown export, whereas literal ``mo.md("...")`` cells are unwrapped to plain Markdown and are unaffected. :func:`convert_admonitions` here applies the same transform outside fenced code blocks only.

**Dropped cells were unreported.** ``inject_outputs`` cannot tell a cell that genuinely produces no output from one whose hash failed to match, so it says nothing either way. Checking the source text instead is decisive, and catches the whole class rather than this one instance: the HTML export lists a hash for every cell in the notebook, so a code fence in the Markdown whose hash is absent from that list is a fence some transform has rewritten. :func:`check_sources_agree` raises on that rather than letting the gap reach the published document.

Usage: ``./go render <notebook.py>``, which is this script with the output defaulted to :func:`~mini.reports.render_path` and the mtime staleness check on. Called directly it is ``uv run scripts/export_report_md.py <notebook.py> [out.md]``. Inlined images are externalized beside the output; a report that publishes through ``mini.reports.report_bundle`` has already written its figures to its ``public/.mini/`` dir, and :func:`localize_links` repoints those links so they resolve from wherever the render lands rather than only from the notebook's own directory. See ``clean_marimo_md.py`` for what the cleanup pass does and for the options this script forwards to it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

from marimo_md_export.export import export_html, export_md, strip_header_from_frontmatter
from marimo_md_export.inject import inject_outputs
from marimo_md_export.models import Cell
from marimo_md_export.parse_html import _extract_session_cells_raw, extract_outputs
from marimo_md_export.parse_md import collect_cells
from marimo_md_export.transform import convert_admonitions as _convert_admonitions
from mini.reports import is_stale, render_path

_SPEC = importlib.util.spec_from_file_location("clean_marimo_md", Path(__file__).with_name("clean_marimo_md.py"))
assert _SPEC and _SPEC.loader
clean_marimo_md = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(clean_marimo_md)

FENCE_RE = re.compile(r"(`{3,}|~{3,})")


def fenced_spans(md: str) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` character offsets of each fenced code block.

    Follows CommonMark's rules for the cases that come up in a Marimo export: a closing fence uses the same character as its opener and is at least as long, and carries no info string. Both matter here, because Marimo widens a cell's fence when the cell's own source contains backticks — the inner ones must not read as the close.
    """
    spans: list[tuple[int, int]] = []
    open_fence: str | None = None
    start = pos = 0

    for line in md.splitlines(keepends=True):
        stripped = line.lstrip()
        m = FENCE_RE.match(stripped)
        if m:
            token = m.group(1)
            if open_fence is None:
                open_fence, start = token, pos
            elif token[0] == open_fence[0] and len(token) >= len(open_fence) and not stripped[len(token) :].strip():
                spans.append((start, pos + len(line)))
                open_fence = None
        pos += len(line)

    if open_fence is not None:  # unterminated fence: treat the rest as code
        spans.append((start, len(md)))
    return spans


def convert_admonitions(md: str) -> str:
    """Convert ``///`` admonitions to ``!!!`` form, leaving fenced code alone."""
    out: list[str] = []
    prev = 0
    for start, end in fenced_spans(md):
        out.append(_convert_admonitions(md[prev:start]))
        out.append(md[start:end])
        prev = end
    out.append(_convert_admonitions(md[prev:]))
    return "".join(out)


def notebook_source_hashes(html: bytes) -> set[str]:
    """Return the MD5 of every cell source in the notebook, per the HTML export.

    The session JSON carries a ``code_hash`` for every cell, including the ones that render nothing; ``extract_outputs`` keeps only the cells that produced output, so it can't serve as the reference set here.
    """
    return {cell["code_hash"] for cell in json.loads(_extract_session_cells_raw(html))}


def check_sources_agree(cells: list[Cell], source_hashes: set[str]) -> None:
    """Raise if a code fence in the Markdown carries a source no notebook cell has.

    Cells are matched to their rendered output by MD5 of the source, so any rewriting of a fence's contents between the two exports costs that cell its output — and, if its code is hidden, the cell itself. Cells that Marimo unwrapped into plain Markdown have no fence and so aren't checked; their text is in the document either way.
    """
    unknown = [cell for cell in cells if cell.source_hash not in source_hashes]
    if not unknown:
        return
    heads = "\n".join(f"    {cell.source.strip().splitlines()[0][:60]}" for cell in unknown)
    raise RuntimeError(
        f"{len(unknown)} code fence(s) in the Markdown export carry a source that no "
        f"notebook cell has, so their rendered output cannot be matched and the cells "
        f"would be dropped silently. Something rewrote the fence contents. Cells:\n{heads}"
    )


MD_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)\)")


def _local_path(src: str) -> str | None:
    """The relative path *src* names, or ``None`` if it doesn't name one at all.

    A ``data:`` URI, an off-repo URL and a bare fragment are all complete as they stand — there is no file for them to be repointed at, and nothing missing when they don't resolve. The ``?v=`` cache stamp the publisher appends is dropped: it exists to defeat a browser cache, and a reader following the link on disk would read it as part of the filename.
    """
    path = src.partition("?")[0]
    return None if not path or path.startswith(("#", "/", "data:")) or "://" in path else path


def _localize(src: str, base: Path, out_dir: Path) -> str | None:
    """*src* rewritten to resolve from *out_dir*, or ``None`` if it isn't a file under *base*."""
    if (path := _local_path(src)) is None or not (target := base / path).is_file():
        return None
    return os.path.relpath(target.resolve(), out_dir.resolve()).replace(os.sep, "/")


def localize_links(md: str, *, base: Path, out_dir: Path) -> tuple[str, list[str]]:
    """Repoint each figure link at the file it names, relative to *out_dir*.

    Only images the report *inlined* pass through :func:`~clean_marimo_md.externalize_images`, which writes them beside the output. Everything a ``report_bundle`` publisher wrote is already a file on disk, so the export references it by a path relative to the notebook — ``public/.mini/<stem>/<name>`` — and the link resolves from the notebook's own directory and nowhere else. Rewriting it against the output's directory is what lets the render live under ``.mini/renders/`` (:func:`~mini.reports.render_path`) with its figures still viewable.

    ``<img>`` tags become Markdown images in the same pass, so every figure in the document reads the same way and its alt text — the point of a text render — is prose rather than an attribute. Returns the document and the srcs that named no file, which stay as they were.
    """
    unresolved: list[str] = []

    def rewrite(src: str) -> str | None:
        """The rewritten src, or ``None`` to leave the link as it stands (noting a local one that resolves nowhere)."""
        if (rel := _localize(src, base, out_dir)) is not None:
            return rel
        # Not beside the notebook. Already beside the output (an inlined image the cleanup
        # pass externalized) is right as it stands; anything else local names no file at all.
        if _local_path(src) is not None and _localize(src, out_dir, out_dir) is None:
            unresolved.append(src)
        return None

    def replace_tag(m: re.Match) -> str:
        attrs = dict(clean_marimo_md.IMG_ATTR_RE.findall(m.group(0)))
        rel = rewrite(attrs.get("src", ""))
        return m.group(0) if rel is None else f"![{clean_marimo_md.image_alt(attrs.get('alt', ''))}]({rel})"

    def replace_md(m: re.Match) -> str:
        rel = rewrite(m.group("src"))
        return m.group(0) if rel is None else f"![{m.group('alt')}]({rel})"

    # Markdown links first: the tag pass writes more of them, and a link this pass has
    # already placed relative to the output must not be resolved a second time.
    md = MD_IMAGE_RE.sub(replace_md, md)
    md = clean_marimo_md.IMG_RE.sub(replace_tag, md)
    return md, unresolved


def export(notebook: Path, *, sandbox: bool = False, timeout: int | None = None) -> str:
    """Export ``notebook`` to Markdown with its rendered outputs injected."""
    md = convert_admonitions(export_md(notebook, sandbox=sandbox, timeout=timeout))
    cells = collect_cells(md)
    html = export_html(notebook, sandbox=sandbox, timeout=timeout)
    check_sources_agree(cells, notebook_source_hashes(html))
    result, warnings = inject_outputs(md, cells, extract_outputs(html))
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return strip_header_from_frontmatter(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path, nargs="?", help="defaults to .mini/renders/<key>.md")
    parser.add_argument("--force", action="store_true", help="re-render even if the output looks up to date")
    parser.add_argument("--raw-output", type=Path, help="also write the export before the cleanup pass")
    parser.add_argument("--assets-dir", type=Path, help="defaults to <output-stem>.assets/ beside the output")
    parser.add_argument("--no-reflow", action="store_true", help="keep the source's line wrapping")
    parser.add_argument("--sandbox", action="store_true", help="run marimo export in an isolated uv environment")
    parser.add_argument("--timeout", type=int, help="seconds to wait for each marimo export subprocess")
    args = parser.parse_args()

    dst = args.output or render_path(args.notebook)
    # Same mtime heuristic the bundle export uses for `--stale-only`. Rendering re-runs the
    # notebook, so a repeat pass over an unedited report costs minutes for a byte-identical file.
    if not args.force and not is_stale(args.notebook, dst):
        print(f"fresh  {dst} (newer than the notebook and its inputs — `--force` re-renders)")
        return

    raw = export(args.notebook, sandbox=args.sandbox, timeout=args.timeout)
    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(raw)

    assets_dir = args.assets_dir or dst.with_name(f"{dst.stem}.assets")
    rel_dir = assets_dir.name if assets_dir.parent == dst.parent else str(assets_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = clean_marimo_md.clean(raw, assets_dir, rel_dir, do_reflow=not args.no_reflow)
    text, unresolved = localize_links(text, base=args.notebook.resolve().parent, out_dir=dst.resolve().parent)
    for src in unresolved:
        print(f"WARNING: figure link names no file, left as it stands: {src}", file=sys.stderr)
    dst.write_text(text)
    print(f"render {args.notebook} -> {dst}")


if __name__ == "__main__":
    main()

"""
Render a document to its outputs: the woven Markdown, the HTML page, and (on request) a PDF.

Outputs land in one directory per document — ``.mini/lit/<key>/`` by default, with ``index.html``, ``index.md``, and the ``_assets/`` the figures were written to — so the same relative URLs work opened from disk, served locally, or published as a bundle the way ``mini.reports`` publishes a Marimo export.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from mini.lit.document import Document, Runner, Woven, parse
from mini.lit.page import page, to_html
from mini.reports import Publisher
from mini.runs import data_root

__all__ = ["render", "Rendered", "to_pdf", "output_dir"]


def output_dir(doc: Path, *, live: bool = False) -> Path:
    """``.mini/lit/<key>/``, where the key is the document's path under ``docs/`` without its suffix (``report.py`` takes its directory's name).

    The live server writes to ``.mini/lit-live/<key>/`` instead: its page carries a reload script and versioned asset URLs, so it is a different artifact from a render, and keeping the trees apart means a ``render`` while the server is up never overwrites the page a browser is watching.
    """
    doc = doc.resolve()
    root = data_root().parent
    try:
        rel = doc.relative_to(root / "docs")
    except ValueError:
        rel = Path(doc.name)
    key = rel.parent if rel.stem == "report" and rel.parent != Path(".") else rel.with_suffix("")
    return data_root() / ("lit-live" if live else "lit") / key


@dataclass
class Rendered:
    doc: Document
    woven: Woven
    html: str
    out_dir: Path
    runner: Runner
    seconds: float  # markdown → html


def render(
    doc: Path | str,
    *,
    out_dir: Path | None = None,
    runner: Runner | None = None,
    live: bool = False,
    extra_body: str = "",
    write_markdown: bool = True,
) -> Rendered:
    """Weave *doc* and write ``index.html`` (and ``index.md``) under *out_dir*.

    *live* is the interactive setting: asset URLs carry a content stamp so a browser shows a re-drawn figure, and a re-drawn figure may replace one of the same name. Pass the previous call's *runner* to re-run only the cells that changed.
    """
    path = Path(doc).resolve()
    out = (out_dir or output_dir(path, live=live)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    publish = Publisher(asset_dir=out / "_assets", link="_assets", strict=not live, versioned=live)
    if runner is None:
        runner = Runner(path, publish=publish)
    else:
        runner.publish = publish
    parsed = parse(path)
    woven = runner.weave(parsed)
    t0 = time.perf_counter()
    body = to_html(woven.markdown)
    html = page(body, title=parsed.title, extra_body=extra_body)
    seconds = time.perf_counter() - t0
    _write(out / "index.html", html)
    if write_markdown:
        _write(out / "index.md", woven.markdown)
    return Rendered(parsed, woven, html, out, runner, seconds)


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")  # per process, so two writers never share a temp file
    tmp.write_text(text)
    tmp.replace(path)


def _chromium() -> str | None:
    for c in (
        os.environ.get("CHROMIUM"),
        "/opt/pw-browsers/chromium",
        "chromium",
        "chromium-browser",
        "google-chrome",
        "chrome",
    ):
        if c and (Path(c).is_file() or shutil.which(c)):
            return c
    return None


def to_pdf(html_path: Path, pdf_path: Path | None = None) -> Path:
    """Print the page to PDF with headless Chromium (the same route the Marimo reports take)."""
    exe = _chromium()
    if exe is None:
        raise RuntimeError("no Chromium found: set $CHROMIUM to the binary")
    pdf_path = pdf_path or html_path.with_suffix(".pdf")
    subprocess.run(
        [
            exe,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return pdf_path

"""
Render a document to its outputs: the woven Markdown, the HTML page, and (on request) a PDF.

Outputs land in one directory per document — ``.mini/lit/<key>/`` by default, with ``index.html``, ``index.md``, and the ``_assets/`` the figures were written to — so the same relative URLs work opened from disk, served locally, or published as a bundle through ``mini.reports``.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from mini.lit.document import Document, Runner, Woven, parse
from mini.lit.page import page, to_html
from mini.reports import Publisher
from mini.runs import data_root

__all__ = ["render", "compose", "Rendered", "to_pdf", "output_dir"]


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

    def write(self, *, markdown: bool = True) -> None:
        """Write ``index.html`` (and ``index.md``) under :attr:`out_dir`."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _write(self.out_dir / "index.html", self.html)
        if markdown:
            _write(self.out_dir / "index.md", self.woven.markdown)


def compose(woven: Woven, *, extra_body: str = "") -> tuple[str, float]:
    """The woven Markdown as a complete page, with how long that took."""
    t0 = time.perf_counter()
    html = page(to_html(woven.markdown), title=woven.doc.title, extra_body=extra_body)
    return html, time.perf_counter() - t0


def render(
    doc: Path | str,
    *,
    out_dir: Path | None = None,
    runner: Runner | None = None,
    live: bool = False,
    extra_body: str = "",
    write: bool = True,
    partial: Callable[[Woven], None] | None = None,
) -> Rendered:
    """Weave *doc* and (with *write*) put ``index.html`` and ``index.md`` under *out_dir*.

    *live* is the interactive setting: asset URLs carry a content stamp so a browser shows a re-drawn figure, and a re-drawn figure may replace one of the same name. Pass the previous call's *runner* to re-run only the cells that changed, and *partial* to be handed the document as it stands before each cell that runs (see :meth:`Runner.weave`).
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
    woven = runner.weave(parsed, partial=partial)
    html, seconds = compose(woven, extra_body=extra_body)
    rendered = Rendered(parsed, woven, html, out, runner, seconds)
    if write:
        rendered.write()
    return rendered


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")  # per process, so two writers never share a temp file
    tmp.write_text(text)
    tmp.replace(path)


CHROMIUM_HINT = (
    "no Chromium found. Install one (`sudo apt-get install chromium`, or `uvx playwright install --with-deps chromium`), "
    "or set $CHROMIUM to the binary"
)


SANDBOX_CHROMIUM = "/opt/pw-browsers/chromium"
"""Where the cloud sandbox keeps its Chromium (the same default as :mod:`mini.report_print`)."""


def _chromium() -> str | None:
    """The first Chromium that exists: ``$CHROMIUM`` (or ``$PLAYWRIGHT_CHROMIUM``, the older spelling), the sandbox's, a browser on ``$PATH``, then Playwright's cache (its download runs only once ``playwright install-deps`` has put the shared libraries in place).

    In the cache we take the headless shell ahead of the full browser: it prints the same PDF, and it links against a smaller set of shared libraries, so it starts in containers where the full build cannot (a missing ``libatk-bridge`` or ``libcups`` leaves the loader unable to start a binary that is sitting right there).
    """
    for c in (
        os.environ.get("CHROMIUM") or os.environ.get("PLAYWRIGHT_CHROMIUM"),
        SANDBOX_CHROMIUM,
        "chromium",
        "chromium-browser",
        "google-chrome",
        "chrome",
    ):
        if c and (Path(c).is_file() or shutil.which(c)):
            return c
    pw_cache = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or Path.home() / ".cache" / "ms-playwright")
    for pattern in ("chromium_headless_shell-*/chrome-*/headless_shell", "chromium-*/chrome-*/chrome"):
        for c in sorted(pw_cache.glob(pattern), reverse=True):
            if c.is_file():
                return str(c)
    return None


def to_pdf(html_path: Path, pdf_path: Path | None = None) -> Path:
    """Print the page to PDF with headless Chromium."""
    exe = _chromium()
    if exe is None:
        raise RuntimeError(CHROMIUM_HINT)
    pdf_path = pdf_path or html_path.with_suffix(".pdf")
    argv = [
        exe,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    done = subprocess.run(argv, capture_output=True, timeout=120)
    if done.returncode != 0:
        # 127 from a binary that is present means the loader could not resolve its shared libraries, so say so: the bare exit code reads like a missing file.
        why = f"{exe} exited {done.returncode}"
        if done.returncode == 127 and Path(exe).is_file():
            why += f" — it exists but could not start, most likely a missing shared library (`ldd {exe} | grep 'not found'`)"
        raise RuntimeError(f"{why}\n{done.stderr.decode(errors='replace').strip()}")
    return pdf_path

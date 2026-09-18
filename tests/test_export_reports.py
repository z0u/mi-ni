"""Tests for the preview-path staleness heuristic — which bundles `--stale-only` re-exports.

The heuristic itself is :func:`mini.reports.is_stale`, shared with the Markdown render; these drive it through the bundle path that names ``index.html`` as the output.
"""

import os
from pathlib import Path

import pytest

from tests.conftest import load_script

export_reports = load_script("export_reports")

_APP = "# title: A report\n"

# Fixed stamps an hour apart, so "newer than" is unambiguous and nothing depends on
# filesystem mtime granularity or on how long the test took to run.
BUNDLE_AT = 1_770_000_000.0
BEFORE, AFTER = BUNDLE_AT - 3600, BUNDLE_AT + 3600


def stamp(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


@pytest.fixture
def report(tmp_path: Path) -> Path:
    """One exported report at `docs/ex-1/`, its bundle newer than every input."""
    (tmp_path / "pyproject.toml").write_text("")
    (docs := tmp_path / "docs" / "ex-1").mkdir(parents=True)
    (nb := docs / "report.py").write_text(_APP)
    (experiment := docs / "experiment.py").write_text("def main(ctx): ...\n")
    (out := tmp_path / ".mini" / "exports" / "ex-1").mkdir(parents=True)
    (out / "index.html").write_text("<html></html>")
    for p in (nb, experiment, docs, out / "index.html"):
        stamp(p, BEFORE)
    stamp(out / "index.html", BUNDLE_AT)
    return nb


def test_a_fresh_bundle_is_not_stale(report):
    assert export_reports.bundle_is_stale(report) is False


def test_a_missing_bundle_is_stale(report):
    (export_reports.export_dir(report) / "index.html").unlink()
    assert export_reports.bundle_is_stale(report) is True


def test_an_edited_report_is_stale(report):
    stamp(report, AFTER)
    assert export_reports.bundle_is_stale(report) is True


def test_an_edited_input_beside_the_report_is_stale(report):
    """The re-run case: new results arrive through `experiment.py` while `report.py` sits still."""
    stamp(report.parent / "experiment.py", AFTER)
    assert export_reports.bundle_is_stale(report) is True


def test_a_deleted_input_is_stale(report):
    """No surviving file carries the news, so the directory's own mtime is what registers it."""
    (report.parent / "experiment.py").unlink()
    stamp(report.parent, AFTER)
    assert export_reports.bundle_is_stale(report) is True


def test_recompiled_bytecode_is_not_an_edit(report):
    """Importing `experiment.py` rewrites its bytecode. Counting that would re-export the report every time anything imported it."""
    (cache := report.parent / "__pycache__").mkdir()
    (cache / "experiment.cpython-313.pyc").write_bytes(b"\x00")
    for p in (cache, cache / "experiment.cpython-313.pyc"):
        stamp(p, AFTER)
    stamp(report.parent, BEFORE)  # a rewrite touches the .pyc, not the directory holding it
    assert export_reports.bundle_is_stale(report) is False


def test_a_literate_script_exports_with_its_markdown_face_and_the_report_styles(tmp_path: Path, monkeypatch):
    """The join between `mini.lit` and the bundle: the woven page declares `index.md` as a rendition and carries `docs/report.css`."""
    from mini.reports import MD_TYPE, alternates

    (tmp_path / "pyproject.toml").write_text("")
    (script := tmp_path / "docs" / "lit" / "report.py").parent.mkdir(parents=True)
    script.write_text('# title: Lit\n\n"""Intro."""\n\nx = 1\n\nf"""x is {x}."""\n')
    (css := tmp_path / "report.css").write_text("main.lit { color: rebeccapurple }")
    monkeypatch.setattr(export_reports, "REPORT_CSS", css)
    (out := tmp_path / ".mini" / "exports" / "lit" / "index.html").parent.mkdir(parents=True)

    html = export_reports._weave(script, out)

    assert out.read_text() == html and "x is 1" in html
    assert alternates(html)[MD_TYPE] == "index.md"
    assert "x is 1" in (out.parent / "index.md").read_text()
    assert "rebeccapurple" in html

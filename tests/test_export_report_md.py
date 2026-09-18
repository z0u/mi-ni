"""The report renderer: a literate script weaves to Markdown with its figures beside it."""

import re
from pathlib import Path

import pytest

from tests.conftest import load_script

export_report_md = load_script("export_report_md")

LIT = (
    '# title: Lit\n\n"""Intro."""\n\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nfig\n\n"""Outro."""\n'
)


def test_a_literate_script_weaves_to_the_render_with_its_figures_beside(tmp_path: Path):
    """The woven Markdown is the render, with figures under `<stem>.assets/` beside it."""
    (script := tmp_path / "docs" / "lit" / "report.py").parent.mkdir(parents=True)
    script.write_text(LIT)
    dst = tmp_path / ".mini" / "renders" / "lit.md"

    export_report_md.weave(script, dst)

    md = dst.read_text()
    assert "Intro." in md and "Outro." in md
    srcs = re.findall(r'src="([^"]+)"', md)
    assert srcs and all(s.startswith("lit.assets/") for s in srcs), srcs
    assert all((dst.parent / s).is_file() for s in srcs)
    assert not (dst.parent / "_assets").exists() and not list(dst.parent.glob(".lit-*"))


def test_a_literate_cell_that_raises_fails_the_render(tmp_path: Path):
    (script := tmp_path / "docs" / "lit" / "report.py").parent.mkdir(parents=True)
    script.write_text('# title: Lit\n\n"""Intro."""\n\nraise ValueError("no data")\n')

    with pytest.raises(SystemExit, match="no data"):
        export_report_md.weave(script, tmp_path / ".mini" / "renders" / "lit.md")

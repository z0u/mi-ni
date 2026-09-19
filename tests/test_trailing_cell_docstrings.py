"""Tests for the stray-prose check — variable docstrings in a literate script that weave as paragraphs."""

from pathlib import Path

from tests.conftest import load_script

check = load_script("trailing_cell_docstrings")


def literate(tmp_path: Path, body: str, name: str = "report.py") -> Path:
    """A literate script (`# title:` header) whose cells and prose are *body*."""
    path = tmp_path / name
    path.write_text("# title: T\n\n" + body)
    return path


def test_a_variable_docstring_is_flagged(tmp_path: Path):
    """Every top-level string weaves as prose, so a docstring hung under a constant is a stray paragraph."""
    script = literate(tmp_path, 'NAMES = ["emb"]\n"""What the slices are called."""\n')

    (finding,) = check.findings_in(script)
    assert finding.line == 4
    assert str(finding).endswith(":4: docstring hangs under an assignment, so it weaves as prose")


def test_prose_after_a_cell_stands_apart_with_a_blank_line(tmp_path: Path):
    """The ordinary shape of a literate script: a cell, a blank line, then the paragraph about its result."""
    script = literate(tmp_path, 'x = compute()\n\n"""x is what it is."""\n\nS = 1\nT = 2\n"""Not this either:"""\n')

    assert [f.line for f in check.findings_in(script)] == [9]


def test_a_docstring_under_a_call_is_prose(tmp_path: Path):
    """Only an assignment can carry a variable docstring; a string after a call is a paragraph about its result."""
    script = literate(tmp_path, 'fig = draw()\nfig\n"""The figure."""\n')

    assert check.findings_in(script) == []


def test_a_plain_module_has_nothing_to_weave(tmp_path: Path):
    """An `experiment.py` beside a report is importable Python, and drops out with nothing to report."""
    path = tmp_path / "experiment.py"
    path.write_text('N = 1\n"""Doc."""\n')

    assert check.findings_in(path) == []


def test_unparseable_files_are_left_to_the_linters(tmp_path: Path):
    broken = literate(tmp_path, "def (:\n", name="broken.py")

    assert check.findings_in(broken) == []


def test_python_files_walks_a_tree(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "report.py").write_text("")
    (tmp_path / "notes.md").write_text("")

    assert check.python_files(tmp_path) == [tmp_path / "sub" / "report.py"]


def test_the_docs_tree_is_clean():
    """The gate itself: every report we publish, checked the way `./go lint` checks it."""
    assert [str(f) for f in (f for p in check.python_files(check.ROOT / "docs") for f in check.findings_in(p))] == []

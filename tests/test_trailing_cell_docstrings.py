"""Tests for the stray-output check — Marimo cells that end on a docstring and publish it."""

from pathlib import Path

import pytest

from tests.conftest import load_script

check = load_script("trailing_cell_docstrings")

HEADER = "import marimo\n\napp = marimo.App()\n\n\n"  # 5 lines, so a cell body starts at line 6


def notebook(tmp_path: Path, body: str, name: str = "report.py") -> Path:
    """A Marimo notebook whose cells are *body*."""
    path = tmp_path / name
    path.write_text(HEADER + body)
    return path


def flagged(path: Path) -> set[str]:
    return {f.cell for f in check.findings_in(path)}


def test_a_setup_cell_ending_on_a_docstring_is_flagged(tmp_path: Path):
    """The case the check exists for: the string lands at the top of the rendered report."""
    nb = notebook(tmp_path, 'with app.setup:\n    NAMES = ["emb"]\n    """What the slices are called."""\n')

    (finding,) = check.findings_in(nb)
    assert (finding.cell, finding.line) == ("setup", 8)


def test_a_bare_none_is_the_fix(tmp_path: Path):
    """`None` as the last statement sends Marimo down its no-output branch."""
    nb = notebook(tmp_path, 'with app.setup:\n    NAMES = ["emb"]\n    """What the slices are called."""\n    None\n')

    assert check.findings_in(nb) == []


def test_a_docstring_mid_cell_is_safe(tmp_path: Path):
    """Only the last statement becomes the output, so an earlier docstring never leaks."""
    nb = notebook(tmp_path, 'with app.setup:\n    A = 1\n    """Doc."""\n    B = 2\n')

    assert check.findings_in(nb) == []


@pytest.mark.parametrize("setup", ["with app.setup:", "with app.setup(hide_code=True):"])
def test_every_spelling_of_the_setup_block(tmp_path: Path, setup: str):
    nb = notebook(tmp_path, f'{setup}\n    A = 1\n    """Doc."""\n')

    assert flagged(nb) == {"setup"}


@pytest.mark.parametrize("ret", ["return", "return (A,)"], ids=["bare", "with-values"])
def test_the_generated_return_does_not_shield_a_cell(tmp_path: Path, ret: str):
    """Marimo strips a cell function's `return` before compiling, so the docstring is still last."""
    nb = notebook(tmp_path, f'@app.cell\ndef _cell():\n    A = 1\n    """Doc."""\n    {ret}\n')

    assert flagged(nb) == {"_cell"}


@pytest.mark.parametrize("decorator", ["@app.cell", "@app.cell(hide_code=True)"])
def test_every_spelling_of_the_cell_decorator(tmp_path: Path, decorator: str):
    nb = notebook(tmp_path, f'{decorator}\ndef _cell():\n    A = 1\n    """Doc."""\n    return\n')

    assert flagged(nb) == {"_cell"}


@pytest.mark.parametrize("decorator", ["@app.function", "@app.class_definition"])
def test_a_function_or_class_body_publishes_nothing(tmp_path: Path, decorator: str):
    """Their body is ordinary local scope, so a trailing string is dead code rather than output."""
    nb = notebook(tmp_path, f'{decorator}\ndef helper():\n    A = 1\n    """Doc."""\n    return A\n')

    assert check.findings_in(nb) == []


def test_a_nested_lookalike_is_ordinary_code(tmp_path: Path):
    """Marimo writes cells at module level; a `with app.setup` inside a function is not one."""
    nb = notebook(tmp_path, 'def build():\n    with app.setup:\n        A = 1\n        """Doc."""\n')

    assert check.findings_in(nb) == []


@pytest.mark.parametrize(
    "last",
    ["fig", "mo.md(f'{n} trials')", "A = 1", "print(A)"],
    ids=["name", "markdown", "assignment", "call"],
)
def test_other_trailing_statements_are_left_alone(tmp_path: Path, last: str):
    """A trailing expression is how a cell shows a figure or some prose — deliberate, and not ours."""
    nb = notebook(tmp_path, f"@app.cell\ndef _cell():\n    A = 1\n    {last}\n    return\n")

    assert check.findings_in(nb) == []


def test_a_plain_module_has_no_cells(tmp_path: Path):
    """An `experiment.py` beside a report is importable Python, and drops out with nothing to report."""
    nb = notebook(tmp_path, 'def main():\n    """Doc."""\n', name="experiment.py")

    assert check.findings_in(nb) == []


def test_unparseable_files_are_left_to_the_linters(tmp_path: Path):
    broken = tmp_path / "broken.py"
    broken.write_text("def (:\n")

    assert check.findings_in(broken) == []


def test_python_files_walks_a_tree(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "report.py").write_text("")
    (tmp_path / "notes.md").write_text("")

    assert check.python_files(tmp_path) == [tmp_path / "sub" / "report.py"]


def test_the_docs_tree_is_clean():
    """The gate itself: every notebook we publish, checked the way `./go lint` checks it."""
    assert [str(f) for f in (f for p in check.python_files(check.ROOT / "docs") for f in check.findings_in(p))] == []

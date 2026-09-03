"""Tests for the propagation check — public cell variables defined without an annotation."""

from pathlib import Path

import pytest

from tests.conftest import load_script

check = load_script("unannotated_cell_vars")

HEADER = "import marimo\n\napp = marimo.App()\n\n\n"


def notebook(tmp_path: Path, body: str, name: str = "report.py") -> Path:
    """A Marimo notebook whose cells are *body*, dedented one level as written here."""
    path = tmp_path / name
    path.write_text(HEADER + body)
    return path


def flagged(path: Path) -> set[str]:
    return {f.name for f in check.findings_in(path)}


def unpacked(path: Path) -> set[str]:
    return {f.name for f in check.findings_in(path) if f.unpacked}


def test_bare_public_assignment_is_flagged(tmp_path: Path):
    """The case the check exists for: a name every downstream cell will receive bare."""
    nb = notebook(tmp_path, "@app.cell\ndef _():\n    trials = load()\n    return (trials,)\n")

    (finding,) = check.findings_in(nb)
    assert (finding.name, finding.line, finding.unpacked) == ("trials", 8, False)  # HEADER is 5 lines


def test_annotated_public_assignment_is_not(tmp_path: Path):
    """One annotation at the definition is what types the whole notebook downstream."""
    nb = notebook(tmp_path, "@app.cell\ndef _():\n    trials: dict[int, dict] = load()\n    return (trials,)\n")

    assert check.findings_in(nb) == []


def test_cell_local_names_are_none_of_our_business(tmp_path: Path):
    """A leading underscore keeps a name inside its cell, so nothing downstream can see its type."""
    nb = notebook(tmp_path, "@app.cell\ndef _():\n    _scratch = load()\n    return\n")

    assert check.findings_in(nb) == []


@pytest.mark.parametrize(
    "statement",
    [
        "def helper(x: int) -> int: return x",  # carries its own types
        "class Record: pass",
        "record['key'] = 1",  # mutates something that exists; defines no cell variable
        "count += 1",
    ],
    ids=["def", "class", "subscript", "augmented"],
)
def test_bindings_with_nothing_to_annotate(tmp_path: Path, statement: str):
    """Only assignments can carry an annotation Marimo would propagate, so only those are asked to."""
    nb = notebook(tmp_path, f"@app.cell\ndef _():\n    {statement}\n    return\n")

    assert check.findings_in(nb) == []


def test_unpacking_is_flagged_separately(tmp_path: Path):
    """`a, b = ...` can't carry an annotation at all, so its fix is to split rather than to annotate."""
    nb = notebook(tmp_path, "@app.cell\ndef _():\n    arrays, metrics = load()\n    return (arrays, metrics)\n")

    assert unpacked(nb) == {"arrays", "metrics"} == flagged(nb)


def test_starred_unpacking_binds_a_name_too(tmp_path: Path):
    nb = notebook(tmp_path, "@app.cell\ndef _():\n    head, *rest = load()\n    return (head, rest)\n")

    assert unpacked(nb) == {"head", "rest"}


def test_chained_assignment_flags_both_targets(tmp_path: Path):
    nb = notebook(tmp_path, "@app.cell\ndef _():\n    lo = hi = 0\n    return (lo, hi)\n")

    assert flagged(nb) == {"lo", "hi"}


def test_a_nested_function_body_is_local_scope(tmp_path: Path):
    """Names inside a helper never reach another cell, whatever they're called."""
    nb = notebook(
        tmp_path,
        "@app.cell\ndef _():\n    def build():\n        rows = []\n        return rows\n\n    return (build,)\n",
    )

    assert check.findings_in(nb) == []


@pytest.mark.parametrize("decorator", ["@app.cell", "@app.cell()", "@app.cell(hide_code=True)"], ids=str)
def test_every_spelling_of_the_cell_decorator(tmp_path: Path, decorator: str):
    nb = notebook(tmp_path, f"{decorator}\ndef _():\n    trials = load()\n    return (trials,)\n")

    assert flagged(nb) == {"trials"}


def test_app_function_is_not_a_cell(tmp_path: Path):
    """`@app.function` defines an ordinary function; its body is local scope, not cell scope."""
    nb = notebook(tmp_path, "@app.function\ndef build():\n    rows = []\n    return rows\n")

    assert check.findings_in(nb) == []


def test_a_plain_module_has_no_cells(tmp_path: Path):
    """`experiment.py` is importable Python beside the notebook, and drops out without a name check."""
    path = tmp_path / "experiment.py"
    path.write_text("def main(ctx):\n    results = ctx.map(train, [1, 2])\n    return results\n")

    assert check.findings_in(path) == []


def test_unparseable_files_are_left_to_the_linters(tmp_path: Path):
    """A syntax error is ruff's finding to report; this check declines to crash over it."""
    path = tmp_path / "broken.py"
    path.write_text("@app.cell\ndef _(:\n")

    assert check.findings_in(path) == []


def test_python_files_walks_a_tree(tmp_path: Path):
    """Collection is by suffix; whether a file is a notebook is the parse's business, not this one's."""
    (tmp_path / "m2" / "ex-1").mkdir(parents=True)
    (tmp_path / "m2" / "ex-1" / "report.py").write_text(HEADER)
    (tmp_path / "m2" / "ex-1" / "experiment.py").write_text("def main(ctx): ...\n")
    (tmp_path / "m2" / "ex-1" / "notes.md").write_text("not python")
    (tmp_path / "index.py").write_text(HEADER)

    assert {p.name for p in check.python_files(tmp_path)} == {"report.py", "experiment.py", "index.py"}

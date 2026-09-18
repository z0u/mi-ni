#!/usr/bin/env python
"""Report docstrings that a report would publish as prose: a Marimo cell's last statement, or a literate script's variable docstring.

Marimo takes a cell's *output* to be the value of its last statement, when that statement is an expression (`_ast/compiler.py`, `if isinstance(final_expr, ast.Expr) and not ends_with_semicolon(code)`). A variable docstring — the triple-quoted string we hang under a constant to say what it holds — is an expression statement, structurally identical to a closing `mo.md("...")`. So a setup cell that ends on one publishes it as a paragraph at the top of the report.

It is invisible everywhere but the rendered page. The source looks like documentation, and `hide_code=True` keeps the cell from showing what produced the stray text. Two reports carried one for weeks before a reader noticed.

The fix is a bare `None` as the cell's last statement, which sends the compiler down its other branch and gives the cell no output. Ruff can't take this over: `B018` is the matching rule, but `docs/*.py` ignores it precisely so variable docstrings are allowed, and it has no notion of which statement is last.

Which cells can leak was settled by exporting a probe notebook and grepping the HTML for each shape:

- `with app.setup` — the cell's last statement, as written.
- `@app.cell` — the last statement before the generated `return`, which Marimo strips whether it is bare or carries values.
- `@app.function` and `@app.class_definition` — never. Their body is an ordinary function or class scope, so a trailing string is dead code rather than an output.

A literate script (`mini.lit`) has the same leak in a different place: *every* top-level string is prose there, so a variable docstring hung under an assignment weaves as a paragraph. It is told from prose proper by where it sits: a docstring starts on the line after its assignment, and prose stands apart from the cell before it with a blank line. That is a convention rather than a guarantee (a docstring set off by a blank line is well-formed prose to every tool, and only shows as a stray paragraph in the render), so the port check's paragraph diff is the second line. The fix there is a comment.
"""

import argparse
import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeIs

from mini.lit import is_literate_script

ROOT = Path(__file__).parent.parent.resolve()

CELL_DECORATOR = "app.cell"
SETUP_CONTEXT = "app.setup"


@dataclass(frozen=True, order=True)
class Finding:
    """One cell that ends on a docstring, and so publishes it."""

    path: Path
    line: int
    cell: (
        str | None
    )  # how to find the cell in the file: "setup", or the cell function's name; None in a literate script

    def __str__(self) -> str:
        where = self.path.relative_to(ROOT) if self.path.is_relative_to(ROOT) else self.path
        what = (
            f"is the output of the {self.cell} cell"
            if self.cell
            else "hangs under an assignment, so it weaves as prose"
        )
        return f"{where.as_posix()}:{self.line}: docstring {what}"


def _is_setup(node: ast.stmt) -> TypeIs[ast.With]:
    """Whether *node* is the `with app.setup:` block, in either of its spellings."""
    if not isinstance(node, ast.With):
        return False
    for item in node.items:
        call = item.context_expr
        target = call.func if isinstance(call, ast.Call) else call  # `app.setup` or `app.setup(...)`
        if ast.unparse(target).endswith(SETUP_CONTEXT):
            return True
    return False


def cells(tree: ast.Module) -> Iterator[tuple[str, list[ast.stmt]]]:
    """Each Marimo cell in *tree*, as its name and the statements Marimo will run.

    Module-level only, because that is where Marimo writes cells; a nested lookalike is ordinary code. The trailing `return` of a `@app.cell` function is dropped here, since Marimo drops it too — with it in the way, no cell would ever end on a docstring and the check would find nothing.
    """
    for node in tree.body:
        if _is_setup(node):
            yield "setup", node.body
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if any(ast.unparse(d).startswith(CELL_DECORATOR) for d in node.decorator_list):
                body = node.body[:-1] if isinstance(node.body[-1], ast.Return) else node.body
                yield node.name, body


def _is_string(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def hung_docstrings(tree: ast.Module) -> Iterator[int]:
    """The line of each top-level string that starts on the line after an assignment, as a variable docstring does."""
    prev: ast.stmt | None = None
    for node in tree.body:
        if (
            _is_string(node)
            and isinstance(prev, ast.Assign | ast.AnnAssign)
            and node.lineno == (prev.end_lineno or prev.lineno) + 1
        ):
            yield node.lineno
        prev = node


def findings_in(path: Path) -> list[Finding]:
    """Every docstring in *path* that renders as output: a Marimo cell's last statement, or a literate script's variable docstring."""
    try:
        tree = ast.parse(path.read_text("utf-8", errors="ignore"), filename=str(path))
    except SyntaxError:
        return []  # not our check's failure to report; ruff and the formatter own that

    if is_literate_script(path):
        return sorted(Finding(path, line, None) for line in hung_docstrings(tree))
    return sorted(Finding(path, body[-1].lineno, name) for name, body in cells(tree) if body and _is_string(body[-1]))


def python_files(root: Path) -> list[Path]:
    """Every `.py` file under *root*, sorted — notebook or not.

    No text match for `marimo.App(` and no notebook filter: a file either has cells or it doesn't, and :func:`findings_in` parses it to find out either way. A plain `experiment.py` beside a report drops out on its own with nothing to report.
    """
    return sorted(p for p in Path(root).rglob("*.py") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, help="notebooks or directories to check (default: docs/)")
    args = ap.parse_args()

    paths = args.paths or [ROOT / "docs"]
    if missing := [p for p in paths if not p.exists()]:
        sys.exit(f"no such path: {', '.join(p.as_posix() for p in missing)}")

    targets = [p for arg in paths for p in (python_files(arg) if arg.is_dir() else [arg])]
    found = sorted(f for path in targets for f in findings_in(path))

    if not found:
        print("✅ No docstring renders as output")
        return

    for finding in found:  # stdout is the worklist, so it stays pipeable
        print(finding)

    print(  # the remedy is commentary, so it goes to stderr and out of the pipe
        f"\n{len(found)} docstring(s) would be published as output."
        " End a Marimo cell with a bare `None` to give it no output; in a literate script, write the docstring as a comment.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()

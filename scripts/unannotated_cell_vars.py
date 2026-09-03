#!/usr/bin/env python
"""Report public Marimo cell variables defined without a type annotation — the propagation check.

Marimo copies a variable's annotation from the cell that *defines* it onto the parameter list of every cell that reads it, and rewrites those signatures on every save (`marimo check --fix`, which the edit hook runs). So one annotation at the definition types the whole notebook downstream, and a missing one leaves every consumer's parameter bare — which is where inference stalls, since a cell function has nothing else to work from at its top. Annotating pays far better than it used to; this names the sites where the payment hasn't been made.

Scoped to *assignments*, which is what keeps it tractable. A public name bound by a `def` or `class` carries its own types already, and Marimo leaves those parameters bare downstream rather than synthesizing a `Callable`, so there is nothing to annotate and nothing to report.

Two kinds of finding, because they have different fixes. A **bare** binding (`x = ...`) becomes `x: T = ...` where it is defined. An **unpacked** one (`a, b = ...`) can't carry an annotation at all — Python has no syntax for it — so the fix is to split the statement, or to accept it. They're counted separately, and only the bare ones set the exit status.

Advisory by design, like `./go dead`: it prints a worklist rather than gating a merge, because the backlog it found on arrival is larger than any one branch should carry. Wire it into the gate once that list is short.

Ruff can't express this rule today: `PYI052` (unannotated-assignment) is stubs-only, and the `ANN` rules cover function signatures rather than assignments.
"""

import argparse
import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

# A cell is `@app.cell` or `@app.cell(...)` on a module-level function; `@app.function`
# defines an ordinary function whose body is local scope, so its assignments aren't cell
# variables and don't propagate anywhere.
CELL_DECORATOR = "app.cell"


@dataclass(frozen=True, order=True)
class Finding:
    """One public cell variable that reaches downstream cells untyped."""

    path: Path
    line: int
    name: str
    unpacked: bool  # bound by tuple/list unpacking, so no annotation is possible

    def __str__(self) -> str:
        # Repo-relative where that's meaningful (the usual case, and what an editor links on),
        # and as given otherwise — a path passed from outside the checkout still has to print.
        where = self.path.relative_to(ROOT) if self.path.is_relative_to(ROOT) else self.path
        return f"{where.as_posix()}:{self.line}: {self.name}{' (unpacked)' if self.unpacked else ''}"


def cell_functions(tree: ast.Module) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    """The cell bodies of a Marimo notebook: module-level functions carrying `@app.cell`.

    Module-level rather than :func:`ast.walk`, because that's where Marimo writes them — a nested lookalike would be ordinary code.
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if any(ast.unparse(d).startswith(CELL_DECORATOR) for d in node.decorator_list):
                yield node


def _bound_names(target: ast.expr) -> list[tuple[ast.Name, bool]]:
    """The names an assignment target binds, each with whether unpacking bound it.

    A `Subscript` or `Attribute` target (`d[k] = ...`, `obj.x = ...`) mutates something that already exists rather than defining a cell variable, so it binds nothing here.
    """
    match target:
        case ast.Name():
            return [(target, False)]
        case ast.Tuple() | ast.List():
            return [(n, True) for e in target.elts for n, _ in _bound_names(e)]
        case ast.Starred():
            return [(n, True) for n, _ in _bound_names(target.value)]
        case _:
            return []


def findings_in(path: Path) -> list[Finding]:
    """Every public cell variable in *path* that an unannotated assignment defines, bare or unpacked.

    A leading underscore marks a name cell-local — Marimo neither exports it nor propagates anything about it — so those are none of this check's business. An `AnnAssign` is the annotated form, and simply isn't an `ast.Assign`, so it drops out of the match below rather than needing a test of its own.
    """
    try:
        tree = ast.parse(path.read_text("utf-8", errors="ignore"), filename=str(path))
    except SyntaxError:
        return []  # not our check's failure to report; ruff and the formatter own that

    return sorted(
        Finding(path, name.lineno, name.id, unpacked)
        for cell in cell_functions(tree)
        for stmt in cell.body  # the cell's own scope: a nested def's locals go nowhere
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
        for name, unpacked in _bound_names(target)
        if not name.id.startswith("_")
    )


def python_files(root: Path) -> list[Path]:
    """Every `.py` file under *root*, sorted — notebook or not.

    Deliberately no notebook filter here, and in particular no text match for `marimo.App(`: this module's own tests carry that string, and so does anything that writes a notebook. A file either has `@app.cell` functions or it doesn't, and :func:`findings_in` has to parse it to find out either way — so the parse is the predicate, and an `experiment.py` (plain, importable Python beside a report) drops out on its own with nothing to report.
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
        print("✅ Every public cell variable is annotated")
        return

    for finding in found:  # stdout is the worklist, so it stays pipeable
        print(finding)

    bare = sum(1 for f in found if not f.unpacked)
    unpacked = len(found) - bare
    print(  # the tally is commentary, so it goes to stderr and out of the pipe
        f"\n{len(found)} public cell variable(s) reach downstream cells untyped"
        f" — {bare} to annotate at the definition"
        f"{f', {unpacked} bound by unpacking (split the statement, or accept it)' if unpacked else ''}.",
        file=sys.stderr,
    )
    sys.exit(1 if bare else 0)


if __name__ == "__main__":
    main()

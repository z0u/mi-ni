#!/usr/bin/env python
"""Report variable docstrings in a literate script, which weave as prose.

Every top-level string in a literate script (`mini.lit`) is prose, so a docstring hung under an assignment weaves as a paragraph. The source looks like documentation, and nothing in the rendered page says where the stray paragraph came from. Ruff can't take this over: `B018` is the matching rule, but a literate script displays a cell's last expression, so the rule is off under `docs/`.

A variable docstring is told from deliberate prose by its position: it starts on the line right after an assignment, where prose stands apart with a blank line. The fix is to write it as a comment.

Exit status is the finding count clamped to 1, so `./go lint` can gate on it; stdout lists the findings, one per line, and the remedy goes to stderr.
"""

import argparse
import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from mini.lit import is_literate_script

ROOT = Path(__file__).parent.parent.resolve()


@dataclass(frozen=True, order=True)
class Finding:
    """One docstring that weaves as prose."""

    path: Path
    line: int

    def __str__(self) -> str:
        where = self.path.relative_to(ROOT) if self.path.is_relative_to(ROOT) else self.path
        return f"{where.as_posix()}:{self.line}: docstring hangs under an assignment, so it weaves as prose"


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
    """Every variable docstring in *path*, when it is a literate script; a plain module has nothing to weave."""
    if not is_literate_script(path):
        return []
    try:
        tree = ast.parse(path.read_text("utf-8", errors="ignore"), filename=str(path))
    except SyntaxError:
        return []  # not our check's failure to report; ruff and the formatter own that
    return sorted(Finding(path, line) for line in hung_docstrings(tree))


def python_files(root: Path) -> list[Path]:
    """Every `.py` file under *root*, sorted — literate or not.

    No filter here: :func:`findings_in` reads the header to tell a literate script from a plain `experiment.py` beside a report, and the plain module drops out with nothing to report.
    """
    return sorted(p for p in Path(root).rglob("*.py") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, help="scripts or directories to check (default: docs/)")
    args = ap.parse_args()

    paths = args.paths or [ROOT / "docs"]
    if missing := [p for p in paths if not p.exists()]:
        sys.exit(f"no such path: {', '.join(p.as_posix() for p in missing)}")

    targets = [p for arg in paths for p in (python_files(arg) if arg.is_dir() else [arg])]
    found = sorted(f for path in targets for f in findings_in(path))

    if not found:
        print("✅ No docstring weaves as prose")
        return

    for finding in found:  # stdout is the worklist, so it stays pipeable
        print(finding)

    print(  # the remedy is commentary, so it goes to stderr and out of the pipe
        f"\n{len(found)} docstring(s) would be published as prose. Write each as a comment instead.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()

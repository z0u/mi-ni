"""
`mini/__init__.py` re-exports lazily, so three declarations describe the same set:
the `TYPE_CHECKING` import block (what static tools see), `_EXPORTS` (what resolves
at runtime), and `__all__` (what ruff and `ty` read, since neither follows a computed
list). Nothing keeps them in step on its own — these tests do.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import mini

INIT = Path(mini.__file__)


def _type_checking_imports() -> dict[str, str]:
    """Name -> module, read out of the `if TYPE_CHECKING:` block without importing it."""
    tree = ast.parse(INIT.read_text())
    return {
        alias.asname or alias.name: node.module
        for block in tree.body
        if isinstance(block, ast.If) and ast.unparse(block.test) == "TYPE_CHECKING"
        for node in ast.walk(block)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }


def test_all_matches_the_lazy_map():
    assert sorted(mini.__all__) == sorted(mini._EXPORTS)


def test_type_checking_block_matches_the_lazy_map():
    """A name static tools can resolve but `__getattr__` can't would fail only at runtime."""
    assert _type_checking_imports() == mini._EXPORTS


@pytest.mark.parametrize("name", sorted(mini.__all__))
def test_every_export_resolves(name):
    assert getattr(mini, name) is not None


def test_unknown_name_raises_attribute_error():
    """The import system relies on this to fall back to `from mini import <submodule>`."""
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        mini.nope  # noqa: B018

    from mini import lineage  # a submodule, not in `__all__`

    assert lineage.__name__ == "mini.lineage"


def test_leaf_module_import_stays_light():
    """
    The point of the exercise: `mini.reports` imports only the standard library, but a
    package `__init__` runs before any submodule, so eager re-exports put `modal` (and
    an environment that has it) behind every `import mini.anything`.
    """
    probe = (
        "import sys; import mini.reports; print(int(any(m == 'modal' or m.startswith('modal.') for m in sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "0", "importing mini.reports pulled in modal"

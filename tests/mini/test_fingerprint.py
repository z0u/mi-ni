"""Key semantics: identity must hold still while evidence tracks the code.

The contract has two sides. *Honesty*: editing anything a task actually depends on — a helper (however it's referenced), a module-level constant, a method — must change the attempt evidence (``code_fp``), or a re-run silently serves stale results. *Stability*: the identity key must be identical across processes, across distinct-but-identical function objects, **and across code edits** — the key is where the task's record, logs, and history live, so an edit must re-run it in place, not orphan it.

Module-level dependencies are exercised with real modules written to disk (the fingerprint reads *source*, so the functions must have files); "editing" is simulated by loading a variant of the module from a sibling directory with the same module name, keeping the task's own source byte-identical.
"""

from __future__ import annotations

import enum
import importlib.util
import sys
from pathlib import Path

import pytest

from mini.memo import task_key, task_key_parts

TASK_ATTR = "import helpers\n\ndef task(x):\n    return helpers.helper(x)\n"
TASK_DEFERRED = "def task(x):\n    from helpers import helper\n\n    return helper(x)\n"
TASK_DEFERRED_MOD = "def task(x):\n    import helpers\n\n    return helpers.helper(x)\n"
TASK_DEFERRED_INDIRECT = "def task(x):\n    from wrapper import wrapped\n\n    return wrapped(x)\n"
WRAPPER = "from helpers import helper\n\ndef wrapped(x):\n    return helper(x) * 2\n"
TASK_NESTED = "from helpers import helper\n\ndef task(xs):\n    inner = lambda v: helper(v)  # noqa: E731\n    return [inner(x) for x in xs]\n"
TASK_METHOD = "from helpers import helper\n\nclass Model:\n    def run(self, x):\n        return helper(x)\n\ndef task(x):\n    return Model().run(x)\n"
TASK_VALUE = "LR = 0.1\n\ndef task(x):\n    return x * LR\n"

HELPER_V1 = "def helper(x):\n    return x + 1\n"
HELPER_V2 = "def helper(x):\n    return x + 2\n"


@pytest.fixture
def load_module(tmp_path: Path):
    """Write and import a module from a per-variant subdir; unimport on teardown."""
    loaded: list[str] = []

    def _load(name: str, source: str, variant: str):
        d = tmp_path / variant
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # so `import helpers` inside a task module resolves
        loaded.append(name)
        spec.loader.exec_module(mod)
        return mod

    yield _load
    for name in loaded:
        sys.modules.pop(name, None)


@pytest.fixture
def deferred_modules(tmp_path: Path, monkeypatch):
    """Write a variant's modules and make them resolvable the way the fingerprint resolves a deferred import: by searching ``sys.path``, without importing.

    One variant is on the path at a time (same module names, different bodies), and the resolver's path cache is cleared between them — within a real process a module name maps to one file, so the cache is only wrong here.
    """
    from mini.memo import _module_file, _module_index

    def _clear() -> None:
        _module_file.cache_clear()
        _module_index.cache_clear()

    def _write(variant: str, **modules: str) -> Path:
        d = tmp_path / variant
        d.mkdir(parents=True, exist_ok=True)
        for name, source in modules.items():
            (d / f"{name}.py").write_text(source)
        monkeypatch.syspath_prepend(d)
        _clear()
        return d

    yield _write
    _clear()


def _deferred_parts(load_module, deferred_modules, task_src: str, variant: str, **modules: str) -> tuple[str, dict]:
    """Fingerprint a task whose body imports *modules* — which are on the path but deliberately never imported, as a driver process would leave them."""
    deferred_modules(variant, **modules)
    return task_key_parts(load_module("tasks", task_src, variant).task, (1,))


@pytest.mark.parametrize(
    "task_src,dep,extra",
    [
        (TASK_DEFERRED, "helpers:helper", {}),
        (TASK_DEFERRED_MOD, "module:helpers", {}),
        (TASK_DEFERRED_INDIRECT, "helpers:helper", {"wrapper": WRAPPER}),
    ],
    ids=["from-import", "module import", "through another module"],
)
def test_deferred_imports_are_tracked(load_module, deferred_modules, task_src: str, dep: str, extra: dict[str, str]):
    """A task that imports project code *inside its body* — the usual way to keep a driver light when the import pulls jax — still depends on that code. Editing it must move the evidence, whether the task imports the helper directly or reaches it through a module that does, or the next wake serves a stale memo hit.

    ``explain`` should name what moved as precisely as the import allowed: the helper itself for a ``from`` import, the whole module for a plain ``import``, where the name reached through it can't be read off the source."""
    key_v1, p_v1 = _deferred_parts(load_module, deferred_modules, task_src, "a", helpers=HELPER_V1, **extra)
    key_v2, p_v2 = _deferred_parts(load_module, deferred_modules, task_src, "b", helpers=HELPER_V2, **extra)
    _, p_copy = _deferred_parts(load_module, deferred_modules, task_src, "c", helpers=HELPER_V1, **extra)
    assert p_v1["code_fp"] != p_v2["code_fp"], "deferred-import edit invisible — stale results would be served"
    assert key_v1 == key_v2, "the edit re-keyed the task — record/logs/history would be orphaned"
    assert p_copy["code_fp"] == p_v1["code_fp"], "identical source must fingerprint identically"
    assert dep in p_v1["deps"], "explain should name what moved"


def test_deferred_library_imports_are_not_tracked(load_module, deferred_modules):
    """Only *project* modules join the evidence: a deferred ``import json`` reaches the stdlib, whose churn must not invalidate anyone's cache."""
    src = "def task(x):\n    import json\n\n    from helpers import helper\n\n    return json.dumps(helper(x))\n"
    _, parts = _deferred_parts(load_module, deferred_modules, src, "a", helpers=HELPER_V1)
    assert "helpers:helper" in parts["deps"]
    assert not [k for k in parts["deps"] if k.startswith(("module:json", "json:"))]


TASK_GHOST = "def task(x):\n    from ghost import helper\n\n    return helper(x)\n"
TASK_GHOST_SUBMODULE = "def task(x):\n    from pkg.ghost import helper\n\n    return helper(x)\n"
TASK_EXTENSION = "def task(x):\n    import math\n\n    return math.sqrt(x)\n"
TASK_NAMESPACE = "def task(x):\n    from nspkg.leaf import go\n\n    return go(x)\n"
TASK_NAMESPACE_GHOST = "def task(x):\n    from nspkg.gone import go\n\n    return go(x)\n"
TASK_OPTIONAL_DEP = (
    "def task(x):\n"
    "    try:\n"
    "        from fastmath import boost\n"
    "    except ImportError:\n"
    "        def boost(v):\n"
    "            return v\n"
    "\n"
    "    return boost(x)\n"
)


def _no_layout(root: Path) -> None:
    """Nothing on disk but the task itself."""


def _project_package(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("")


def _namespace_package(root: Path) -> None:
    """PEP 420: a directory with no ``__init__.py``, with a real submodule under it."""
    (root / "nspkg").mkdir(parents=True, exist_ok=True)
    (root / "nspkg" / "leaf.py").write_text("def go(x):\n    return x + 1\n")


@pytest.mark.parametrize(
    "task_src,layout,warns,dep",
    [
        (TASK_GHOST, _no_layout, "'ghost'", None),
        (TASK_GHOST_SUBMODULE, _project_package, "'pkg.ghost'", None),
        (TASK_EXTENSION, _no_layout, None, None),
        (TASK_NAMESPACE, _namespace_package, None, "nspkg.leaf:go"),
        (TASK_NAMESPACE_GHOST, _namespace_package, "'nspkg.gone'", None),
    ],
    ids=[
        "unresolvable module",
        "missing submodule of a project package",
        "extension module",
        "local namespace package",
        "missing submodule of a namespace package",
    ],
)
def test_only_genuine_holes_warn(load_module, deferred_modules, tmp_path, caplog, task_src, layout, warns, dep):
    """A module the walk can't find is the one case where "not project code" is a lie: it is skipped exactly as the stdlib is, so the task depends on nothing and its record caches forever — a stale result served for the life of the module. The source genuinely isn't there to read, so all that's left is to say so.

    Which makes the silent cases the hard part, because a path search finds nothing for them either. ``math`` is C; a PEP 420 namespace package is a bare directory; a wheel may ship no source at its root. Warning on those would fire on every task that imports the stdlib, which is how a warning stops being read — so the question is who *claims* the name, not whether a file turned up. The silence is bought by naming the portion rather than its root, so a genuine hole *under* a namespace package is still said out loud, and the submodule below one still reaches the evidence.

    The realistic hole is ``from sca.thing import x`` after ``thing`` moved: the root package is project code and resolves fine, so only the leaf is missing — which is what makes it easy to miss.
    """
    layout(tmp_path / "a")
    with caplog.at_level("WARNING", logger="mini.memo"):
        _, parts = _deferred_parts(load_module, deferred_modules, task_src, "a")
    if warns:
        assert [r for r in caplog.records if warns in r.message], "the hole said nothing about itself"
        assert not [k for k in parts["deps"] if warns.strip("'") in k], "the hazard: the import joined no evidence"
    else:
        assert not caplog.records
    if dep:
        assert dep in parts["deps"], "the submodule still has to be tracked"


def test_deliberately_absent_imports_warn_and_say_so(load_module, deferred_modules, caplog):
    """A known false positive, pinned rather than fixed: an optional dependency behind ``try/except ImportError`` warns whenever it is absent, which is the case the code was written to handle.

    Telling it apart from a real hole needs the AST context that says "this import is guarded", and the walk reaches a task body as bytecode, where the ``try`` has become a jump — the same reason ``if TYPE_CHECKING:`` imports of uninstalled packages warn. What the message can do is admit the possibility, so a reader isn't sent looking for a broken install that was never broken. If a later change does learn to tell the two apart, this test should flip."""
    with caplog.at_level("WARNING", logger="mini.memo"):
        _deferred_parts(load_module, deferred_modules, TASK_OPTIONAL_DEP, "a")
    warnings = [r for r in caplog.records if "'fastmath'" in r.message]
    assert warnings, "the shape still warns"
    assert "meant to be absent" in warnings[0].message, "…and the message has to allow that it is fine"


def test_installed_extension_packages_are_not_holes(monkeypatch):
    """The same silence has to cover wheels that ship no source at their root — a C extension like ``_xxhash``, or a namespace package. Installed metadata is what says so, since the path search can't."""
    from mini import memo

    monkeypatch.setattr(memo, "_installed_roots", lambda: frozenset({"speedy"}))
    assert not memo._should_have_resolved("speedy")
    assert not memo._should_have_resolved("speedy.core")
    assert memo._should_have_resolved("ghost")


BIG_HELPERS = HELPER_V1 + "\n\ndef unrelated(x):\n    return x * 1000\n\nUNUSED = 'a' * 500\n"
TASK_ALIAS = "def task(x):\n    from pkg import helpers as h\n\n    return h.helper(x)\n"
TASK_ALIAS_BARE = "def task(x):\n    from pkg import helpers as h\n\n    return h.helper(x) + len(dir(h))\n"


@pytest.mark.parametrize("task_src", [TASK_DEFERRED, TASK_DEFERRED_INDIRECT], ids=["direct", "through a wrapper"])
def test_deferred_evidence_is_the_symbol_not_the_module(load_module, deferred_modules, task_src: str):
    """The evidence should be the *helper*, not the file it lives in.

    A deferred import is deferred because the module is expensive, which tends to mean it's also big — so taking it whole makes a task depend on hundreds of lines it never calls, and every unrelated edit re-runs it (on ex-2.1.5, half the ``sca`` package for a fn whose real references were a twentieth of that). Editing a sibling function must leave the fingerprint alone."""
    _, p_v1 = _deferred_parts(load_module, deferred_modules, task_src, "a", helpers=BIG_HELPERS, wrapper=WRAPPER)
    _, p_edited = _deferred_parts(
        load_module,
        deferred_modules,
        task_src,
        "b",
        helpers=BIG_HELPERS.replace("x * 1000", "x * 2000").replace("'a' * 500", "'b' * 500"),
        wrapper=WRAPPER,
    )
    assert p_v1["code_fp"] == p_edited["code_fp"], "an unrelated sibling edit re-ran the task for nothing"
    assert not [k for k in p_v1["deps"] if k.startswith("module:")], "the module was taken whole"


def test_module_alias_narrows_to_the_attributes_reached(load_module, deferred_modules, tmp_path):
    """``from pkg import helpers as h`` then ``h.helper()`` names one function, so that's what the evidence should be — while ``h`` passed somewhere else could reach anything in the module, and has to take it whole."""

    def parts(task_src: str, helpers: str, variant: str) -> dict:
        d = tmp_path / variant
        (d / "pkg").mkdir(parents=True, exist_ok=True)
        (d / "pkg" / "__init__.py").write_text("")
        (d / "pkg" / "helpers.py").write_text(helpers)
        deferred_modules(variant)
        return task_key_parts(load_module("tasks", task_src, variant).task, (1,))[1]

    narrowed = parts(TASK_ALIAS, BIG_HELPERS, "a")
    assert "pkg.helpers:helper" in narrowed["deps"]
    assert "module:pkg.helpers" not in narrowed["deps"]
    assert narrowed["code_fp"] == parts(TASK_ALIAS, BIG_HELPERS.replace("x * 1000", "x * 2000"), "b")["code_fp"]

    # The same alias handed to `dir()`: what it reaches is no longer readable, so
    # the whole module counts and the same sibling edit *does* invalidate.
    whole = parts(TASK_ALIAS_BARE, BIG_HELPERS, "c")
    assert "module:pkg.helpers" in whole["deps"]
    assert whole["code_fp"] != parts(TASK_ALIAS_BARE, BIG_HELPERS.replace("x * 1000", "x * 2000"), "d")["code_fp"]


TASK_PKG_DEFERRED = "def task(x):\n    from pkg.mod import helper\n\n    return helper(x)\n"
TASK_PKG_TOPLEVEL = "from pkg.mod import helper\n\ndef task(x):\n    return helper(x)\n"


@pytest.fixture
def pkg_parts(load_module, deferred_modules, tmp_path):
    """Fingerprint a task that reaches ``pkg.mod:helper``, against a given ``pkg/__init__.py``."""

    def _parts(task_src: str, init: str, variant: str, mod: str = HELPER_V1) -> tuple[str, dict]:
        d = tmp_path / variant
        (d / "pkg").mkdir(parents=True, exist_ok=True)
        (d / "pkg" / "__init__.py").write_text(init)
        (d / "pkg" / "mod.py").write_text(mod)
        deferred_modules(variant)
        for name in [n for n in sys.modules if n == "pkg" or n.startswith("pkg.")]:
            del sys.modules[name]  # the next variant's `pkg` is a different file
        return task_key_parts(load_module("tasks", task_src, variant).task, (1,))

    return _parts


@pytest.mark.parametrize(
    "task_src", [TASK_PKG_DEFERRED, TASK_PKG_TOPLEVEL], ids=["deferred import", "module-scope import"]
)
def test_package_init_is_evidence_however_the_helper_is_reached(pkg_parts, task_src: str):
    """A package's ``__init__`` runs before the module under it does, and can change what the task computes (``sca/__init__.py`` sets ``XLA_FLAGS``), so it belongs in the evidence and editing it must re-run.

    Which is easy for the deferred walk, resolving a dotted name it can read the chain off. The module-scope import reaches the helper as an *object*, and the chain has to come from its ``__module__`` instead — otherwise the same edit to the same file invalidates one task and not the other, purely by where the import was written.
    """
    grown = "VERSION = 1\n\ndef unrelated():\n    return 99\n"
    key_v1, p_v1 = pkg_parts(task_src, "VERSION = 1\n", "a")
    key_v2, p_v2 = pkg_parts(task_src, grown, "b")
    _, p_copy = pkg_parts(task_src, "VERSION = 1\n", "c")
    assert "module:pkg" in p_v1["deps"], "the parent package left no trace — edits to it can't re-run the task"
    assert p_v1["code_fp"] != p_v2["code_fp"], "package-init edit invisible — stale results would be served"
    assert key_v1 == key_v2, "the edit re-keyed the task — record/logs/history would be orphaned"
    assert p_copy["code_fp"] == p_v1["code_fp"], "identical source must fingerprint identically"


# Import-time behavior in the *defining* module rather than a package above it: the
# helper's own source is untouched by an edit to the line beside it.
MOD_PRELUDE = "import os\n\nos.environ.setdefault('PKG_MODE', '{mode}')\n\n" + HELPER_V1


@pytest.mark.parametrize(
    "task_src", [TASK_PKG_DEFERRED, TASK_PKG_TOPLEVEL], ids=["deferred import", "module-scope import"]
)
def test_a_modules_import_time_statements_are_evidence(pkg_parts, task_src: str):
    """The other half of what runs before a helper does: statements at the top of its *own* module. Editing one leaves the helper's source byte-identical, so only the module's prelude entry can carry it — and the task's behavior really does change, since that's where an ``os.environ.setdefault`` lands."""
    _, p_v1 = pkg_parts(task_src, "", "a", mod=MOD_PRELUDE.format(mode="fast"))
    _, p_v2 = pkg_parts(task_src, "", "b", mod=MOD_PRELUDE.format(mode="slow"))
    assert "pkg.mod:<module>" in p_v1["deps"], "the module's import-time statements left no trace"
    assert p_v1["code_fp"] != p_v2["code_fp"], "import-time edit invisible — stale results would be served"


def _key_and_parts(load_module, task_src: str, helper_src: str, variant: str) -> tuple[str, dict]:
    load_module("helpers", helper_src, variant)
    tasks = load_module("tasks", task_src, variant)
    return task_key_parts(tasks.task, (1,))


@pytest.mark.parametrize(
    "task_src",
    [TASK_ATTR, TASK_NESTED, TASK_METHOD],
    ids=["module-attr call", "nested-code reference", "via a method"],
)
def test_helper_edits_move_evidence_not_identity(load_module, task_src: str):
    """Editing a helper must change the task's evidence (so it re-runs) whether it's called by bare name, as a module attribute (``helpers.helper``), from inside a nested lambda / comprehension, or from a method of a class the task uses — while the *key* stays put, so the re-run lands on the same record. An identical copy must produce identical evidence (no path or object identity in the fingerprint)."""
    key_v1, p_v1 = _key_and_parts(load_module, task_src, HELPER_V1, "a")
    key_v2, p_v2 = _key_and_parts(load_module, task_src, HELPER_V2, "b")
    key_copy, p_copy = _key_and_parts(load_module, task_src, HELPER_V1, "c")
    assert p_v1["code_fp"] != p_v2["code_fp"], "helper edit invisible to evidence — stale results would be served"
    assert key_v1 == key_v2, "helper edit re-keyed the task — record/logs/history would be orphaned"
    assert (key_copy, p_copy["code_fp"]) == (key_v1, p_v1["code_fp"]), "identical source must fingerprint identically"


DOCUMENTED = '"""Colour helpers."""\n\n\ndef helper(x):\n    """Add one to *x*."""\n    return x + 1  # the increment\n'


@pytest.mark.parametrize(
    "reworded,moves",
    [
        (DOCUMENTED.replace("Add one to *x*.", "Increment *x* by one."), False),
        (DOCUMENTED.replace("Colour helpers.", "Color helpers."), False),
        (DOCUMENTED.replace("# the increment", "# add one"), True),
        (DOCUMENTED.replace("return x + 1", "return x + 2"), True),
    ],
    ids=["function docstring", "module docstring", "comment", "code"],
)
def test_docstring_edits_do_not_invalidate(load_module, reworded: str, moves: bool):
    """A documentation pass over ``src/`` should not cost a sweep. A docstring is the one piece of source with no behavior behind it, so rewording one — on the helper or on its module — must leave the evidence where it is.

    Comments are deliberately still evidence: a changed comment usually rides along with changed code, so the over-invalidating bias stays where it earns its keep, and this is a carve-out for docstrings alone."""
    _, base = _key_and_parts(load_module, TASK_ATTR, DOCUMENTED, "a")
    _, edited = _key_and_parts(load_module, TASK_ATTR, reworded, "b")
    assert (base["code_fp"] != edited["code_fp"]) is moves


def test_module_level_value_edits_invalidate(load_module):
    """A module-level constant a task reads (``LR``) is part of its behavior: editing the value must change the evidence, exactly like editing code."""
    _, p_v1 = task_key_parts(load_module("tasks", TASK_VALUE, "a").task, (1,))
    _, p_v2 = task_key_parts(load_module("tasks", TASK_VALUE.replace("0.1", "0.2"), "b").task, (1,))
    _, p_copy = task_key_parts(load_module("tasks", TASK_VALUE, "c").task, (1,))
    assert p_v1["code_fp"] != p_v2["code_fp"]
    assert p_v1["code_fp"] == p_copy["code_fp"]


def _make_callback(delta: int):
    """A fresh function object per call — same source, different identity."""
    if delta == 1:

        def cb(x):
            return x + 1
    else:

        def cb(x):
            return x + 2

    return cb


def test_callable_inputs_key_by_source_not_identity():
    """A function passed as *data* is an input, so it fingerprints into the key by its source: two fresh objects of the same source coincide (a repr would embed a memory address and relaunch the task every wake), while a different body diverges — a new input, a new cell."""

    def apply(f, x):
        return f(x)

    assert task_key(apply, (_make_callback(1), 5)) == task_key(apply, (_make_callback(1), 5))
    assert task_key(apply, (_make_callback(1), 5)) != task_key(apply, (_make_callback(2), 5))


class _Color(enum.Enum):
    RED = 1
    BLUE = 2


def test_enum_and_path_inputs_are_stable_and_distinct():
    def t(v):
        return v

    assert task_key(t, (_Color.RED,)) == task_key(t, (_Color.RED,))
    assert task_key(t, (_Color.RED,)) != task_key(t, (_Color.BLUE,))
    assert task_key(t, (Path("/a/b"),)) == task_key(t, (Path("/a/b"),))
    assert task_key(t, (Path("/a/b"),)) != task_key(t, (Path("/a/c"),))


def test_self_referential_global_does_not_recurse(load_module):
    """A module-level container holding the task itself (a registry pattern) must not send the collector into infinite recursion."""
    src = "CALLBACKS = []\n\ndef task(x):\n    return len(CALLBACKS) + x\n\nCALLBACKS.append(task)\n"
    mod = load_module("tasks", src, "a")
    assert task_key_parts(mod.task, (1,))  # completes; no RecursionError


def test_parts_split_code_from_inputs(load_module):
    """``explain`` relies on the parts: same code + different inputs moves only ``input_fp`` (a different cell); an edited helper moves only ``code_fp`` (and names the dep)."""
    load_module("helpers", HELPER_V1, "a")
    tasks = load_module("tasks", TASK_ATTR, "a")
    k1, p1 = task_key_parts(tasks.task, (1,))
    k2, p2 = task_key_parts(tasks.task, (2,))
    assert p1["code_fp"] == p2["code_fp"] and p1["input_fp"] != p2["input_fp"]
    assert k1 != k2  # inputs are identity

    load_module("helpers", HELPER_V2, "b")
    tasks_b = load_module("tasks", TASK_ATTR, "b")
    k3, p3 = task_key_parts(tasks_b.task, (1,))
    assert p3["input_fp"] == p1["input_fp"] and p3["code_fp"] != p1["code_fp"]
    assert k3 == k1  # code is evidence, not identity
    changed = [k for k in p1["deps"] if p3["deps"].get(k) != p1["deps"][k]]
    assert changed == ["helper"]  # the diff names exactly the dependency that moved


def test_version_is_evidence_not_identity():
    """``version=`` forces a re-run *in place*: it moves the evidence while the key stays put, so the bump lands as a new attempt on the same record."""

    def t(x):
        return x

    k1, p1 = task_key_parts(t, (1,), version="v1")
    k2, p2 = task_key_parts(t, (1,), version="v2")
    assert k1 == k2
    assert (p1.get("version"), p2.get("version")) == ("v1", "v2")


def test_repr_fallback_warns_about_unstable_inputs(caplog):
    """Inputs with no stable encoding (an object whose repr embeds its address) can never cache — that's a silent money-burner, so it must warn."""

    class Opaque:
        __slots__ = ()

    def t(o):
        return o

    with caplog.at_level("WARNING", logger="mini.memo"):
        task_key(t, (Opaque(),))
    assert any("never be a cache hit" in r.message for r in caplog.records)

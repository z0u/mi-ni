"""Key semantics: identity must hold still while evidence tracks the code.

The contract has two sides. *Honesty*: editing anything a task actually depends
on — a helper (however it's referenced), a module-level constant, a method —
must change the attempt evidence (``code_fp``), or a re-run silently serves
stale results. *Stability*: the identity key must be identical across processes,
across distinct-but-identical function objects, **and across code edits** — the
key is where the task's record, logs, and history live, so an edit must re-run
it in place, not orphan it.

Module-level dependencies are exercised with real modules written to disk (the
fingerprint reads *source*, so the functions must have files); "editing" is
simulated by loading a variant of the module from a sibling directory with the
same module name, keeping the task's own source byte-identical.
"""

from __future__ import annotations

import enum
import importlib.util
import sys
from pathlib import Path

import pytest

from mini.memo import task_key, task_key_parts

TASK_ATTR = "import helpers\n\ndef task(x):\n    return helpers.helper(x)\n"
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
    """Editing a helper must change the task's evidence (so it re-runs) whether
    it's called by bare name, as a module attribute (``helpers.helper``), from
    inside a nested lambda / comprehension, or from a method of a class the task
    uses — while the *key* stays put, so the re-run lands on the same record.
    An identical copy must produce identical evidence (no path or object
    identity in the fingerprint)."""
    key_v1, p_v1 = _key_and_parts(load_module, task_src, HELPER_V1, "a")
    key_v2, p_v2 = _key_and_parts(load_module, task_src, HELPER_V2, "b")
    key_copy, p_copy = _key_and_parts(load_module, task_src, HELPER_V1, "c")
    assert p_v1["code_fp"] != p_v2["code_fp"], "helper edit invisible to evidence — stale results would be served"
    assert key_v1 == key_v2, "helper edit re-keyed the task — record/logs/history would be orphaned"
    assert (key_copy, p_copy["code_fp"]) == (key_v1, p_v1["code_fp"]), "identical source must fingerprint identically"


def test_module_level_value_edits_invalidate(load_module):
    """A module-level constant a task reads (``LR``) is part of its behavior:
    editing the value must change the evidence, exactly like editing code."""
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
    """A function passed as *data* is an input, so it fingerprints into the key
    by its source: two fresh objects of the same source coincide (a repr would
    embed a memory address and relaunch the task every wake), while a different
    body diverges — a new input, a new cell."""

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
    """A module-level container holding the task itself (a registry pattern) must
    not send the collector into infinite recursion."""
    src = "CALLBACKS = []\n\ndef task(x):\n    return len(CALLBACKS) + x\n\nCALLBACKS.append(task)\n"
    mod = load_module("tasks", src, "a")
    assert task_key_parts(mod.task, (1,))  # completes; no RecursionError


def test_parts_split_code_from_inputs(load_module):
    """``explain`` relies on the parts: same code + different inputs moves only
    ``input_fp`` (a different cell); an edited helper moves only ``code_fp``
    (and names the dep)."""
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
    """``version=`` forces a re-run *in place*: it moves the evidence while the
    key stays put, so the bump lands as a new attempt on the same record."""

    def t(x):
        return x

    k1, p1 = task_key_parts(t, (1,), version="v1")
    k2, p2 = task_key_parts(t, (1,), version="v2")
    assert k1 == k2
    assert (p1.get("version"), p2.get("version")) == ("v1", "v2")


def test_repr_fallback_warns_about_unstable_inputs(caplog):
    """Inputs with no stable encoding (an object whose repr embeds its address)
    can never cache — that's a silent money-burner, so it must warn."""

    class Opaque:
        __slots__ = ()

    def t(o):
        return o

    with caplog.at_level("WARNING", logger="mini.memo"):
        task_key(t, (Opaque(),))
    assert any("never be a cache hit" in r.message for r in caplog.records)

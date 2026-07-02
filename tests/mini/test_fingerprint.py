"""Fingerprint semantics: what invalidates a memo key, and what must not.

The contract has two sides. *Honesty*: editing anything a task actually depends
on — a helper (however it's referenced), a module-level constant, a method —
must change the key, or a re-run silently serves stale results. *Stability*: the
key must be identical across processes and across distinct-but-identical
function objects, or a task relaunches on every wake.

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

from mini.memo import fingerprint, fingerprint_parts

TASK_ATTR = 'import helpers\n\ndef task(x):\n    return helpers.helper(x)\n'
TASK_NESTED = 'from helpers import helper\n\ndef task(xs):\n    inner = lambda v: helper(v)  # noqa: E731\n    return [inner(x) for x in xs]\n'
TASK_METHOD = 'from helpers import helper\n\nclass Model:\n    def run(self, x):\n        return helper(x)\n\ndef task(x):\n    return Model().run(x)\n'
TASK_VALUE = 'LR = 0.1\n\ndef task(x):\n    return x * LR\n'

HELPER_V1 = 'def helper(x):\n    return x + 1\n'
HELPER_V2 = 'def helper(x):\n    return x + 2\n'


@pytest.fixture
def load_module(tmp_path: Path):
    """Write and import a module from a per-variant subdir; unimport on teardown."""
    loaded: list[str] = []

    def _load(name: str, source: str, variant: str):
        d = tmp_path / variant
        d.mkdir(parents=True, exist_ok=True)
        path = d / f'{name}.py'
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


def _fp_with_helper(load_module, task_src: str, helper_src: str, variant: str) -> str:
    load_module('helpers', helper_src, variant)
    tasks = load_module('tasks', task_src, variant)
    return fingerprint(tasks.task, (1,))


@pytest.mark.parametrize(
    'task_src',
    [TASK_ATTR, TASK_NESTED, TASK_METHOD],
    ids=['module-attr call', 'nested-code reference', 'via a method'],
)
def test_helper_edits_invalidate_however_referenced(load_module, task_src: str):
    """Editing a helper must re-key the task whether it's called by bare name,
    as a module attribute (``helpers.helper``), from inside a nested lambda /
    comprehension, or from a method of a class the task uses. The task's own
    source is byte-identical across variants — only the helper differs — and an
    identical copy must produce an identical key (no path or object identity in
    the fingerprint)."""
    fp_v1 = _fp_with_helper(load_module, task_src, HELPER_V1, 'a')
    fp_v2 = _fp_with_helper(load_module, task_src, HELPER_V2, 'b')
    fp_v1_copy = _fp_with_helper(load_module, task_src, HELPER_V1, 'c')
    assert fp_v1 != fp_v2, 'editing the helper did not change the key — stale results would be served'
    assert fp_v1 == fp_v1_copy, 'identical source produced different keys — the task could never cache'


def test_module_level_value_edits_invalidate(load_module):
    """A module-level constant a task reads (``LR``) is part of its behavior:
    editing the value must re-key the task, exactly like editing code."""
    fp_v1 = fingerprint(load_module('tasks', TASK_VALUE, 'a').task, (1,))
    fp_v2 = fingerprint(load_module('tasks', TASK_VALUE.replace('0.1', '0.2'), 'b').task, (1,))
    fp_v1_copy = fingerprint(load_module('tasks', TASK_VALUE, 'c').task, (1,))
    assert fp_v1 != fp_v2
    assert fp_v1 == fp_v1_copy


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
    """A function passed as *data* must fingerprint by its source: two fresh
    objects of the same source coincide (a repr would embed a memory address and
    relaunch the task every wake), while a different body diverges."""

    def apply(f, x):
        return f(x)

    assert fingerprint(apply, (_make_callback(1), 5)) == fingerprint(apply, (_make_callback(1), 5))
    assert fingerprint(apply, (_make_callback(1), 5)) != fingerprint(apply, (_make_callback(2), 5))


class _Color(enum.Enum):
    RED = 1
    BLUE = 2


def test_enum_and_path_inputs_are_stable_and_distinct():
    def t(v):
        return v

    assert fingerprint(t, (_Color.RED,)) == fingerprint(t, (_Color.RED,))
    assert fingerprint(t, (_Color.RED,)) != fingerprint(t, (_Color.BLUE,))
    assert fingerprint(t, (Path('/a/b'),)) == fingerprint(t, (Path('/a/b'),))
    assert fingerprint(t, (Path('/a/b'),)) != fingerprint(t, (Path('/a/c'),))


def test_self_referential_global_does_not_recurse(load_module):
    """A module-level container holding the task itself (a registry pattern) must
    not send the collector into infinite recursion."""
    src = 'CALLBACKS = []\n\ndef task(x):\n    return len(CALLBACKS) + x\n\nCALLBACKS.append(task)\n'
    mod = load_module('tasks', src, 'a')
    assert fingerprint(mod.task, (1,))  # completes; no RecursionError


def test_fingerprint_parts_split_code_from_inputs(load_module):
    """``explain`` relies on the parts: same code + different inputs moves only
    ``input_fp``; an edited helper moves only ``code_fp`` (and names the dep)."""
    load_module('helpers', HELPER_V1, 'a')
    tasks = load_module('tasks', TASK_ATTR, 'a')
    _, p1 = fingerprint_parts(tasks.task, (1,))
    _, p2 = fingerprint_parts(tasks.task, (2,))
    assert p1['code_fp'] == p2['code_fp'] and p1['input_fp'] != p2['input_fp']

    load_module('helpers', HELPER_V2, 'b')
    tasks_b = load_module('tasks', TASK_ATTR, 'b')
    _, p3 = fingerprint_parts(tasks_b.task, (1,))
    assert p3['input_fp'] == p1['input_fp'] and p3['code_fp'] != p1['code_fp']
    changed = [k for k in p1['deps'] if p3['deps'].get(k) != p1['deps'][k]]
    assert changed == ['helper']  # the diff names exactly the dependency that moved


def test_repr_fallback_warns_about_unstable_inputs(caplog):
    """Inputs with no stable encoding (an object whose repr embeds its address)
    can never cache — that's a silent money-burner, so it must warn."""

    class Opaque:
        __slots__ = ()

    def t(o):
        return o

    with caplog.at_level('WARNING', logger='mini.memo'):
        fingerprint(t, (Opaque(),))
    assert any('never be a cache hit' in r.message for r in caplog.records)

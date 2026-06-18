"""Tests for memoized multi-step orchestration (ctx.run/ctx.map + tick).

Task functions are *local* so cloudpickle serializes them by value; the
orchestration ``main`` runs in-process in the driver.
"""

from __future__ import annotations

import time
from pathlib import Path

from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus
from mini.memo import MemoStore, fingerprint
from mini.orchestration import retry, tick
from mini.runs import RunState


def _setup(name: str, main, tmp_path: Path) -> tuple[Experiment, LocalApparatus]:
    """An experiment (no compute) plus the apparatus it's run on (injected)."""
    return Experiment(name=name, main=main), LocalApparatus(name, data_dir=tmp_path / name)


def _drive(exp: Experiment, app: LocalApparatus, timeout: float = 30.0):
    """Re-run the orchestration each 'wake' until it completes (mirrors the agent loop)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, payload = tick(exp, app)
        if done:
            return payload
        time.sleep(0.1)
    raise AssertionError(f'orchestration did not complete: {payload}')


def test_multistep_dependency(tmp_path: Path):
    def prep():
        return {'vocab': 7}

    def train(lr, vocab):
        return {'lr': lr, 'vocab': vocab}

    def main(ctx):
        meta = ctx.run(prep)  # sweep configs depend on prep's output
        return ctx.map(train, [(lr, meta['vocab']) for lr in (0.1, 0.2)])

    assert _drive(*_setup('dep', main, tmp_path)) == [{'lr': 0.1, 'vocab': 7}, {'lr': 0.2, 'vocab': 7}]


def test_prep_runs_once_across_wakes(tmp_path: Path):
    def prep():
        from mini import get_data_dir

        f = get_data_dir() / 'prep_count'
        n = int(f.read_text()) if f.exists() else 0
        f.write_text(str(n + 1))
        return n + 1

    def train(x):
        return x * 2

    def main(ctx):
        ctx.run(prep)
        return ctx.map(train, [(1,), (2,)])

    exp, app = _setup('once', main, tmp_path)
    _drive(exp, app)
    tick(exp, app)  # extra wakes after completion
    tick(exp, app)
    assert (tmp_path / 'once' / 'prep_count').read_text() == '1'  # prep memoized, ran once


def test_failed_is_terminal_until_retry(tmp_path: Path):
    """A thrown task settles FAILED and does *not* auto-relaunch; an explicit
    ``retry`` resets it so the next drive reruns just that task and completes."""

    def train(x):
        from mini import get_data_dir

        f = get_data_dir() / f'att_{x}'
        n = int(f.read_text()) if f.exists() else 0
        f.write_text(str(n + 1))
        if x == 2 and n == 0:  # fails on the first attempt only
            raise RuntimeError('transient')
        return x * 10

    def main(ctx):
        return ctx.map(train, [(1,), (2,), (3,)])

    exp, app = _setup('crash', main, tmp_path)
    store = app.memo_store()

    # Drive until task 2 settles FAILED (siblings DONE), then keep ticking: FAILED
    # is terminal, so it is never relaunched and the map stays blocked.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        tick(exp, app)
        if any(r.get('state') == RunState.FAILED for r in store.records()):
            break
        time.sleep(0.1)
    for _ in range(3):
        tick(exp, app)
        time.sleep(0.1)
    states = {r['key']: r.get('state') for r in store.records()}
    assert sum(s == RunState.FAILED for s in states.values()) == 1  # not relaunched
    assert (tmp_path / 'crash' / 'att_2').read_text() == '1'  # threw exactly once

    # Explicit retry heals: reset the failed task, then drive to completion.
    assert len(retry(store)) == 1  # one FAILED task reset
    assert _drive(exp, app) == [10, 20, 30]
    assert (tmp_path / 'crash' / 'att_2').read_text() == '2'  # ran again on retry
    assert (tmp_path / 'crash' / 'att_1').read_text() == '1'  # siblings untouched


def test_single_map_sugar(tmp_path: Path):
    def sq(x):
        return x * x

    exp = Experiment(name='sugar', fn=sq, configs=[(2,), (3,)])
    app = LocalApparatus('sugar', data_dir=tmp_path / 'sugar')
    assert _drive(exp, app) == [4, 9]


def test_metrics_recorded_on_task(tmp_path: Path):
    def t(x):
        from mini import emit_metrics

        emit_metrics(v=float(x))
        return x

    def main(ctx):
        return ctx.map(t, [(5,)])

    _drive(*_setup('met', main, tmp_path))
    recs = MemoStore(tmp_path / 'met').records()
    assert any(r.get('metrics', {}).get('v') == 5.0 for r in recs)


def test_version_busts_cache(tmp_path: Path):
    def t(x):
        return x

    def main_v1(ctx):
        return ctx.map(t, [(1,)], version='v1')

    def main_v2(ctx):
        return ctx.map(t, [(1,)], version='v2')

    _drive(*_setup('ver', main_v1, tmp_path))
    _drive(Experiment(name='ver', main=main_v2), LocalApparatus('ver', data_dir=tmp_path / 'ver'))
    assert len({r['key'] for r in MemoStore(tmp_path / 'ver').records()}) == 2  # distinct keys per version


def test_per_step_apparatus_uses_its_hooks(tmp_path: Path):
    """``on=`` routes a step to a different apparatus — here proven via its hooks."""

    def mark_default():
        from mini import get_data_dir

        (get_data_dir() / 'default_hook').touch()

    def mark_gpu():
        from mini import get_data_dir

        (get_data_dir() / 'gpu_hook').touch()

    def task(x):
        return x

    data_dir = tmp_path / 'perstep'
    default = LocalApparatus('perstep', data_dir=data_dir).before_each(mark_default)
    gpu = LocalApparatus('perstep', data_dir=data_dir).before_each(mark_gpu)

    def main(ctx):
        return ctx.map(task, [(1,)], on=gpu)

    assert _drive(Experiment(name='perstep', main=main), default) == [1]
    assert (data_dir / 'gpu_hook').exists()  # the on= apparatus's hook ran
    assert not (data_dir / 'default_hook').exists()  # the tick default's did not


def test_ctx_spawns_via_the_apparatus(tmp_path: Path):
    """``ctx`` launches tasks through ``apparatus.spawn_tasks`` — the seam the
    Modal backend implements — and batches a map's fan-out into one call. Proven
    by routing to an apparatus that runs tasks *synchronously in-process* instead
    of spawning: if Ctx bypassed the seam, the drive would time out."""
    from mini._taskworker import run_task
    from mini.local_apparatus import LocalApparatus

    batches: list[int] = []

    class InlineApparatus(LocalApparatus):
        def spawn_tasks(self, store, batch):
            batches.append(len(batch))  # record fan-out width
            for key, fn, args, hooks in batch:
                store.write_call(key, fn, args, hooks)
                run_task(store.data_dir, key)  # run now, in-process — no subprocess

    def task(x):
        return x * 3

    app = InlineApparatus('inline', data_dir=tmp_path / 'inline')
    exp = Experiment(name='inline', main=lambda ctx: ctx.map(task, [(2,), (5,)]))
    assert _drive(exp, app) == [6, 15]
    assert batches == [2]  # both tasks launched in a single batched spawn


def test_fingerprint_is_deterministic_and_input_sensitive():
    def fn(x):
        return x

    assert fingerprint(fn, (1,)) == fingerprint(fn, (1,))
    assert fingerprint(fn, (1,)) != fingerprint(fn, (2,))

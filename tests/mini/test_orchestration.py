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
from mini.orchestration import tick


def _exp(name: str, main, tmp_path: Path) -> Experiment:
    return Experiment(name=name, main=main, apparatus=LocalApparatus(name, data_dir=tmp_path / name))


def _drive(exp: Experiment, timeout: float = 30.0):
    """Re-run the orchestration each 'wake' until it completes (mirrors the agent loop)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, payload = tick(exp)
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

    assert _drive(_exp('dep', main, tmp_path)) == [{'lr': 0.1, 'vocab': 7}, {'lr': 0.2, 'vocab': 7}]


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

    exp = _exp('once', main, tmp_path)
    _drive(exp)
    tick(exp)  # extra wakes after completion
    tick(exp)
    assert (tmp_path / 'once' / 'prep_count').read_text() == '1'  # prep memoized, ran once


def test_crash_recovers_on_rerun(tmp_path: Path):
    def train(x):
        from mini import get_data_dir

        f = get_data_dir() / f'att_{x}'
        n = int(f.read_text()) if f.exists() else 0
        f.write_text(str(n + 1))
        if x == 2 and n == 0:
            raise RuntimeError('transient')
        return x * 10

    def main(ctx):
        return ctx.map(train, [(1,), (2,), (3,)])

    exp = _exp('crash', main, tmp_path)
    assert _drive(exp) == [10, 20, 30]
    assert (tmp_path / 'crash' / 'att_2').read_text() == '2'  # the failed one re-ran
    assert (tmp_path / 'crash' / 'att_1').read_text() == '1'  # its siblings did not


def test_single_map_sugar(tmp_path: Path):
    def sq(x):
        return x * x

    exp = Experiment(
        name='sugar', fn=sq, configs=[(2,), (3,)], apparatus=LocalApparatus('sugar', data_dir=tmp_path / 'sugar')
    )
    assert _drive(exp) == [4, 9]


def test_metrics_recorded_on_task(tmp_path: Path):
    def t(x):
        from mini import emit_metrics

        emit_metrics(v=float(x))
        return x

    def main(ctx):
        return ctx.map(t, [(5,)])

    _drive(_exp('met', main, tmp_path))
    recs = MemoStore(tmp_path / 'met').records()
    assert any(r.get('metrics', {}).get('v') == 5.0 for r in recs)


def test_version_busts_cache(tmp_path: Path):
    def t(x):
        return x

    def main_v1(ctx):
        return ctx.map(t, [(1,)], version='v1')

    def main_v2(ctx):
        return ctx.map(t, [(1,)], version='v2')

    _drive(_exp('ver', main_v1, tmp_path))
    _drive(Experiment(name='ver', main=main_v2, apparatus=LocalApparatus('ver', data_dir=tmp_path / 'ver')))
    assert len({r['key'] for r in MemoStore(tmp_path / 'ver').records()}) == 2  # distinct keys per version


def test_fingerprint_is_deterministic_and_input_sensitive():
    def fn(x):
        return x

    assert fingerprint(fn, (1,)) == fingerprint(fn, (1,))
    assert fingerprint(fn, (1,)) != fingerprint(fn, (2,))

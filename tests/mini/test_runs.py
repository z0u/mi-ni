"""Tests for detached runs: submit / poll / gather / retry, with durable state.

Jobs are defined as *local* functions so cloudpickle serializes them by value —
the detached worker subprocess reconstructs them without importing this module.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mini.local_apparatus import LocalApparatus
from mini.runs import RunState, latest_run, open_run


def _wait(run, timeout: float = 20.0):
    """Poll the durable state until the run settles (mirrors how an agent waits)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if run.done():
            return
        time.sleep(0.05)
    raise AssertionError(f'run did not settle: {[(s.job_id, str(s.state)) for s in run.status()]}')


def test_submit_poll_gather(tmp_path: Path):
    def job(x):
        return x * 2

    app = LocalApparatus('t', max_workers=3, data_dir=tmp_path / 't')
    run = app.submit(job, [1, 2, 3])
    _wait(run)
    assert run.state() == RunState.DONE
    assert run.results() == [2, 4, 6]


def test_results_preserve_input_order(tmp_path: Path):
    def job(x):
        import time as _t

        _t.sleep(0.05 * (3 - x))  # later inputs finish first
        return x

    app = LocalApparatus('order', max_workers=3, data_dir=tmp_path / 'order')
    run = app.submit(job, [1, 2, 3])
    _wait(run)
    assert run.results() == [1, 2, 3]


def test_failure_is_durable(tmp_path: Path):
    def job(x):
        if x == 2:
            raise ValueError('boom')
        return x

    app = LocalApparatus('f', max_workers=3, data_dir=tmp_path / 'f')
    run = app.submit(job, [1, 2, 3])
    _wait(run)

    assert run.state() == RunState.FAILED
    by_id = {s.job_id: s for s in run.status()}
    assert by_id['0'].state == RunState.DONE
    assert by_id['2'].state == RunState.DONE
    assert by_id['1'].state == RunState.FAILED
    assert by_id['1'].error and 'boom' in by_id['1'].error
    assert 'ValueError: boom' in run.logs('1')
    with pytest.raises(RuntimeError, match='not done'):
        run.results()


def test_metrics_surface_in_status(tmp_path: Path):
    def job(x):
        from mini import emit_metrics, emit_progress

        emit_progress(1, 1, message='done')
        emit_metrics(loss=float(x), lr=0.1)
        return x

    app = LocalApparatus('m', max_workers=2, data_dir=tmp_path / 'm')
    run = app.submit(job, [7])
    _wait(run)
    (status,) = run.status()
    assert status.metrics == {'loss': 7.0, 'lr': 0.1}


def test_reopen_in_fresh_handle(tmp_path: Path):
    def job(x):
        return x + 100

    app = LocalApparatus('r', max_workers=2, data_dir=tmp_path / 'r')
    run = app.submit(job, [1, 2])
    _wait(run)

    # A "fresh process" reconstructs the handle from the id alone.
    reopened = app.reopen(run.id)
    assert reopened.state() == RunState.DONE
    assert reopened.results() == [101, 102]


def test_discovery_by_name(tmp_path: Path):
    def job(x):
        return x

    app = LocalApparatus('disco', max_workers=1, data_dir=tmp_path / 'disco')
    run = app.submit(job, [1])
    _wait(run)

    found = latest_run('disco', data_root=tmp_path)
    assert found is not None
    assert found.id == run.id
    # ...and via the id with an explicit root
    assert open_run(run.id, data_root=tmp_path).state() == RunState.DONE


def test_retry_reruns_only_failed(tmp_path: Path):
    # Fail on the first attempt, succeed on the second (a durable per-job counter).
    def job(x):
        from mini import get_data_dir

        marker = get_data_dir() / f'attempts_{x}'
        n = int(marker.read_text()) if marker.exists() else 0
        marker.write_text(str(n + 1))
        if x == 2 and n == 0:
            raise RuntimeError('transient')
        return x * 10

    app = LocalApparatus('retry', max_workers=3, data_dir=tmp_path / 'retry')
    run = app.submit(job, [1, 2, 3])
    _wait(run)
    assert run.state() == RunState.FAILED

    run.retry(failed_only=True)
    _wait(run)
    assert run.state() == RunState.DONE
    assert run.results() == [10, 20, 30]
    # Jobs that already succeeded were not re-run.
    assert (tmp_path / 'retry' / 'attempts_1').read_text() == '1'
    assert (tmp_path / 'retry' / 'attempts_2').read_text() == '2'


def test_empty_submit(tmp_path: Path):
    def job(x):
        return x

    app = LocalApparatus('empty', max_workers=1, data_dir=tmp_path / 'empty')
    run = app.submit(job, [])
    _wait(run)
    assert run.results() == []


def test_load_and_launch_experiment_file(tmp_path: Path):
    from mini.experiment import load_experiment

    exp_file = tmp_path / 'exp.py'
    exp_file.write_text(
        'from mini import Experiment\n'
        'def go(x):\n'
        '    return x + 1\n'
        'experiment = Experiment(name="filexp", fn=go, configs=[(1,), (2,)])\n'
    )
    app = LocalApparatus('filexp', data_dir=tmp_path / 'filexp')
    run = load_experiment(exp_file).submit(app)
    _wait(run)
    assert run.results() == [2, 3]


def test_emit_metrics_outside_context_is_noop():
    from mini import emit_metrics

    emit_metrics(loss=1.0)  # must not raise

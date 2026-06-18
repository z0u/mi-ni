"""CLI inspect/cancel commands over the memo store.

`ls`/`status` surface memo-orchestration experiments (state lives in the memo
store, addressed by name); `cancel` stops in-flight tasks. Commands are driven
against ``.mini`` under a tmp cwd, so DATA_ROOT (a relative path) resolves there.
"""

from __future__ import annotations

import argparse
import os
import textwrap
import time
from pathlib import Path

import pytest

from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus
from mini.orchestration import tick
from mini.runs import RunState


def _drive(exp: Experiment, app: LocalApparatus, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, _ = tick(exp, app)
        if done:
            return
        time.sleep(0.1)
    raise AssertionError('orchestration did not complete')


def test_ls_and_status_surface_memo_experiments(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # DATA_ROOT='.mini' resolves under here

    def train(x):
        return x * 2

    exp = Experiment(name='cli', main=lambda ctx: ctx.map(train, [(1,), (2,)]))
    _drive(exp, LocalApparatus('cli'))  # default data_dir → .mini/cli

    from mini.__main__ import cmd_ls, cmd_status

    cmd_ls(argparse.Namespace())
    ls_out = capsys.readouterr().out
    assert 'cli' in ls_out and 'tasks' in ls_out  # discovered via the memo store

    cmd_status(argparse.Namespace(name='cli', app='local'))
    status_out = capsys.readouterr().out
    assert 'train-' in status_out and '2 tasks' in status_out  # per-task memo records


def test_cancel_stops_running_task(tmp_path: Path):
    def slow(x):
        import time

        time.sleep(30)  # long enough that only a cancel ends it within the test
        return x

    app = LocalApparatus('cancelexp', data_dir=tmp_path / 'cancelexp')
    tick(Experiment(name='cancelexp', main=lambda ctx: ctx.map(slow, [(1,)])), app)  # launch + suspend

    store = app.memo_store()
    (rec,) = store.records()
    pid = rec['pid']  # recorded synchronously at spawn
    assert pid and RunState(rec['state']) == RunState.RUNNING

    assert app.cancel(store) == [rec['key']]
    assert all(RunState(r['state']) == RunState.CANCELLED for r in store.records())

    # the worker really took the SIGTERM (reap it to confirm + avoid a zombie)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            break
        time.sleep(0.05)
    else:
        raise AssertionError('worker did not exit after cancel')


def test_retry_cli_heals_failed_task(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # DATA_ROOT + the experiment file resolve under here
    exp_file = tmp_path / 'retrycli.py'
    exp_file.write_text(
        textwrap.dedent("""
        from mini import Experiment, get_data_dir
        def flaky(x):
            f = get_data_dir() / 'attempts'
            n = int(f.read_text()) if f.exists() else 0
            f.write_text(str(n + 1))
            if n == 0:  # fail on the first attempt only
                raise RuntimeError('boom once')
            return x
        experiment = Experiment(name='retrycli', main=lambda ctx: ctx.map(flaky, [(7,)]))
        """)
    )
    from mini.__main__ import cmd_retry, cmd_run

    def ns():  # run/retry share flags; --watch drives synchronously to settle
        return argparse.Namespace(path=str(exp_file), watch=True, poll=0.05, app='local', workers=1, key=None)

    with pytest.raises(SystemExit):  # FAILED is terminal — watch surfaces it and exits 1
        cmd_run(ns())
    capsys.readouterr()  # drop the failure output

    cmd_retry(ns())  # resets the failed task, then the rerun (attempt 2) succeeds
    out = capsys.readouterr().out
    assert 'retrying 1 task' in out and '✓ complete' in out

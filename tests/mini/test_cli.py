"""CLI inspect commands span both state models (run/job *and* memo).

`mini ls` / `mini status` historically only saw run/job runs (`index.json`);
these assert they now also surface `mini run` orchestration experiments, whose
state lives in the memo store. Commands are driven against ``.mini`` under a
tmp cwd, so DATA_ROOT (a relative path) resolves there.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus
from mini.orchestration import tick


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

    cmd_status(argparse.Namespace(ref='cli'))
    status_out = capsys.readouterr().out
    assert 'train-' in status_out and '2 tasks' in status_out  # per-task memo records

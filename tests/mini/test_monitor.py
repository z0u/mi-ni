"""Tests for the drive-to-completion watcher (``mini run --watch``).

Uses a real ``LocalApparatus`` so tasks run in detached subprocesses — the same
durable-records path the watcher polls. A quiet console keeps Rich off the test
output. See test_orchestration.py for the single-tick ``_drive`` counterpart.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus
from mini.monitor import ExperimentFailed, drive_and_watch


def _quiet() -> Console:
    return Console(file=io.StringIO())


def _watch(exp: Experiment, app: LocalApparatus):
    return drive_and_watch(exp, app, poll=0.05, console=_quiet())


def test_drives_multistep_to_completion(tmp_path: Path):
    def prep():
        return {'vocab': 11}

    def train(lr, vocab):
        return {'lr': lr, 'vocab': vocab}

    def main(ctx):
        meta = ctx.run(prep)  # second stage depends on the first
        return ctx.map(train, [(lr, meta['vocab']) for lr in (0.1, 0.2)])

    app = LocalApparatus('watch_ok', max_workers=2, data_dir=tmp_path / 'watch_ok')
    payload = _watch(Experiment(name='watch_ok', main=main), app)
    assert payload == [{'lr': 0.1, 'vocab': 11}, {'lr': 0.2, 'vocab': 11}]


def test_raises_on_failure_without_relaunching(tmp_path: Path):
    def boom(x):
        raise ValueError(f'boom {x}')

    app = LocalApparatus('watch_fail', max_workers=2, data_dir=tmp_path / 'watch_fail')
    exp = Experiment(name='watch_fail', main=lambda ctx: ctx.map(boom, [(1,), (2,)]))
    with pytest.raises(ExperimentFailed) as exc:
        _watch(exp, app)
    assert len(exc.value.failed) == 2  # both surfaced, not busy-looped

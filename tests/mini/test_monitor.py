"""Tests for the drive-to-completion watcher (``mini run --watch``).

Uses a real ``LocalApparatus`` so tasks run in detached subprocesses — the same
durable-records path the watcher polls. A quiet console keeps Rich off the test
output. See test_orchestration.py for the single-tick ``_drive`` counterpart.
"""

from __future__ import annotations

import io
import os
import signal
import threading
import time
from contextlib import suppress
from pathlib import Path

import pytest
from rich.console import Console

from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus
from mini.monitor import drive_and_watch, watch
from mini.orchestration import TaskFailed, tick
from mini.runs import RunState


def _quiet() -> Console:
    return Console(file=io.StringIO())


def _sleeper(x):
    import time

    time.sleep(60)  # long enough that only a kill ends it within the test
    return x


def _watch(exp: Experiment, app: LocalApparatus):
    return drive_and_watch(exp, app, poll=0.05, console=_quiet())


def test_drives_multistep_to_completion(tmp_path: Path):
    def prep():
        return {"vocab": 11}

    def train(lr, vocab):
        return {"lr": lr, "vocab": vocab}

    def main(ctx):
        meta = ctx.run(prep)  # second stage depends on the first
        return ctx.map(train, [0.1, 0.2], [meta["vocab"]] * 2)

    app = LocalApparatus("watch_ok", max_workers=2, data_dir=tmp_path / "watch_ok")
    payload = _watch(Experiment(name="watch_ok", main=main), app)
    assert payload == [{"lr": 0.1, "vocab": 11}, {"lr": 0.2, "vocab": 11}]


def _times_ten(x):
    return x * 10


def test_watch_observes_a_run_it_did_not_launch(tmp_path: Path):
    """Read-only ``watch``: another caller ``tick``s to launch the detached work;
    ``watch`` only polls the durable records and returns once they settle. It takes
    no experiment, so it structurally *can't* relaunch — the read-only invariant."""
    app = LocalApparatus("watch_ro", max_workers=2, data_dir=tmp_path / "watch_ro")
    exp = Experiment(name="watch_ro", main=lambda ctx: ctx.map(_times_ten, [1, 2]))
    tick(exp, app)  # launch the single stage detached, then suspend — like a separate process

    records = watch(app, poll=0.05, console=_quiet())
    assert all(RunState(r["state"]) == RunState.DONE for r in records)
    store = app.memo_store()
    assert sorted(store.result(r["key"]) for r in records) == [10, 20]


def test_raises_on_failure_without_relaunching(tmp_path: Path):
    def boom(x):
        raise ValueError(f"boom {x}")

    app = LocalApparatus("watch_fail", max_workers=2, data_dir=tmp_path / "watch_fail")
    exp = Experiment(name="watch_fail", main=lambda ctx: ctx.map(boom, [1, 2]))
    with pytest.raises(ExceptionGroup) as exc:
        _watch(exp, app)
    assert len(exc.value.exceptions) == 2  # both surfaced together, not busy-looped
    assert all(isinstance(e, TaskFailed) for e in exc.value.exceptions)


def test_reap_dead_settles_a_killed_worker(tmp_path: Path):
    """A worker hard-killed mid-run (no FAILED written) is detected as dead and
    settled FAILED, so it can't masquerade as RUNNING forever."""
    app = LocalApparatus("reap", data_dir=tmp_path / "reap")
    tick(Experiment(name="reap", main=lambda ctx: ctx.map(_sleeper, [1])), app)  # launch + suspend
    store = app.memo_store()
    (rec,) = store.records()
    pid = rec["pid"]
    assert RunState(rec["state"]) == RunState.RUNNING
    assert app.reap_dead(store) == []  # a live worker is left alone

    os.killpg(pid, signal.SIGKILL)  # crash it without letting it write a result
    deadline = time.monotonic() + 10  # wait until truly gone (a zombie counts as dead)
    while time.monotonic() < deadline and app._is_task_alive(rec):
        time.sleep(0.05)
    assert not app._is_task_alive(rec)

    assert app.reap_dead(store) == [rec["key"]]
    (settled,) = store.records()
    assert RunState(settled["state"]) == RunState.FAILED and "vanished" in settled["error"]
    assert app.reap_dead(store) == []  # idempotent — it's terminal now
    with suppress(ChildProcessError):
        os.waitpid(pid, 0)  # reap the zombie we created


def test_drive_stops_on_cancelled_instead_of_spinning(tmp_path: Path):
    """A CANCELLED task is terminal: ``drive_and_watch`` must stop (raise), not
    busy-loop ticking a DAG that can never progress (nothing RUNNING, no FAILED)."""
    app = LocalApparatus("watch_cancel", data_dir=tmp_path / "watch_cancel")
    exp = Experiment(name="watch_cancel", main=lambda ctx: ctx.map(_sleeper, [1]))
    tick(exp, app)  # launch detached
    store = app.memo_store()

    pid = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (recs := store.records()) and recs[0].get("pid") and recs[0].get("state") == RunState.RUNNING:
            pid = recs[0]["pid"]
            break
        time.sleep(0.05)
    assert pid, "worker never started"

    assert app.cancel(store)  # marks CANCELLED + SIGTERMs the worker
    with pytest.raises(ExceptionGroup) as exc:
        _watch(exp, app)  # must not hang
    failure = exc.value.exceptions[0]
    assert isinstance(failure, TaskFailed)
    assert failure.state == RunState.CANCELLED
    with suppress(ChildProcessError):
        os.waitpid(pid, 0)


def test_watch_surfaces_a_killed_worker(tmp_path: Path):
    """The wedge fix end-to-end: a worker killed *while watching* settles FAILED
    via ``reap_dead``, so the drain raises instead of waiting on it forever."""
    app = LocalApparatus("watch_killed", data_dir=tmp_path / "watch_killed")
    exp = Experiment(name="watch_killed", main=lambda ctx: ctx.map(_sleeper, [1]))

    captured: dict[str, BaseException] = {}

    def run() -> None:
        try:
            _watch(exp, app)
        except BaseException as e:  # noqa: BLE001 — record whatever the watch raised
            captured["exc"] = e

    watcher = threading.Thread(target=run)
    watcher.start()

    store = app.memo_store()  # wait for the detached worker to be RUNNING with a pid
    pid = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (recs := store.records()) and recs[0].get("pid") and recs[0].get("state") == RunState.RUNNING:
            pid = recs[0]["pid"]
            break
        time.sleep(0.05)
    assert pid, "worker never started"

    os.killpg(pid, signal.SIGKILL)
    watcher.join(timeout=15)
    assert not watcher.is_alive(), "watch wedged on the dead worker"
    assert isinstance(captured.get("exc"), ExceptionGroup)
    with suppress(ChildProcessError):
        os.waitpid(pid, 0)

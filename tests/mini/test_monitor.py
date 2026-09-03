"""Tests for the drive-to-completion watcher (``mini run --watch``).

Uses a real ``LocalApparatus`` so tasks run in detached subprocesses — the same durable-records path the watcher polls. A quiet console keeps Rich off the test output. See test_orchestration.py for the single-tick ``_drive`` counterpart.
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
    """Read-only ``watch``: another caller ``tick``s to launch the detached work; ``watch`` only polls the durable records and returns once they settle. It takes no experiment, so it structurally *can't* relaunch — the read-only invariant."""
    app = LocalApparatus("watch_ro", max_workers=2, data_dir=tmp_path / "watch_ro")
    exp = Experiment(name="watch_ro", main=lambda ctx: ctx.map(_times_ten, [1, 2]))
    tick(exp, app)  # launch the single stage detached, then suspend — like a separate process

    records, outcome, reason = watch(app, poll=0.05, console=_quiet())
    assert (outcome, reason) == ("settled", None)
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
    """A worker hard-killed mid-run (no FAILED written) is detected as dead and settled FAILED, so it can't masquerade as RUNNING forever."""
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


def _running_rec(key: str, **extra) -> dict:
    """A fabricated live RUNNING record: our own (alive) pid keeps reap_dead off it, and a fresh heartbeat + env mark it as truly started (not queued)."""
    return {
        "key": key,
        "state": "running",
        "fn": "train",
        "pid": os.getpid(),
        "env": {"host": "worker.test"},
        "heartbeat_at": time.time(),
        **extra,
    }


def _flip(store, key: str, fields: dict, delay: float) -> threading.Thread:
    t = threading.Timer(delay, lambda: store.records_backend.merge(key, fields))
    t.start()
    return t


def test_watch_wakes_on_a_failure_mid_stage(tmp_path: Path):
    """The watchdog-fired scenario: one cell settles FAILED while its siblings run on. ``watch`` must return *immediately* with an attention outcome — not sit until the whole stage settles (or the caller's command times out)."""
    app = LocalApparatus("watch_wake", data_dir=tmp_path / "watch_wake")
    store = app.memo_store()
    store.records_backend.merge("t-healthy", _running_rec("t-healthy"))
    store.records_backend.merge("t-doomed", _running_rec("t-doomed"))
    _flip(store, "t-doomed", {"state": "failed", "error": "WatchdogStall: wedged"}, delay=0.3)

    start = time.monotonic()
    current, outcome, reason = watch(app, poll=0.05, console=_quiet())
    assert outcome == "attention" and reason == "t-doomed settled failed"
    assert time.monotonic() - start < 5  # woke on the event, not on the sibling finishing


def test_watch_ignores_terminal_tasks_from_before_the_watch(tmp_path: Path):
    """A run deliberately advanced past a failed cell must still be watchable: only failures that *happen during* the watch trigger attention, so a pre-existing terminal record doesn't wake every watcher immediately."""
    app = LocalApparatus("watch_pre", data_dir=tmp_path / "watch_pre")
    store = app.memo_store()
    store.records_backend.merge("t-old-fail", {"key": "t-old-fail", "state": "failed", "error": "boom"})
    store.records_backend.merge("t-live", _running_rec("t-live"))
    _flip(store, "t-live", {"state": "done"}, delay=0.3)

    current, outcome, reason = watch(app, poll=0.05, console=_quiet())
    assert (outcome, reason) == ("settled", None)  # ran through to settle, no early wake
    states = {r["key"]: RunState(r["state"]) for r in current}
    assert states == {"t-old-fail": RunState.FAILED, "t-live": RunState.DONE}


def test_watch_wakes_on_a_wedged_worker(tmp_path: Path):
    """The wedge signature — heartbeat fresh, step frozen past the threshold — must wake the watcher (outcome ``attention``) instead of blocking until some timeout: the worker may burn GPU forever without ever settling."""
    from mini.runs import STALE_HEARTBEAT_S

    app = LocalApparatus("watch_wedge", data_dir=tmp_path / "watch_wedge")
    store = app.memo_store()
    now = time.time()
    store.records_backend.merge(
        "t-wedged",
        _running_rec("t-wedged", started_at=now - 900, progress_at=now - STALE_HEARTBEAT_S - 100, step=42, total=100),
    )

    current, outcome, reason = watch(app, poll=0.05, console=_quiet())
    assert outcome == "attention" and reason is not None
    assert "t-wedged" in reason and "no step progress" in reason


def test_watch_timeout_returns_with_work_in_flight(tmp_path: Path):
    app = LocalApparatus("watch_to", data_dir=tmp_path / "watch_to")
    store = app.memo_store()
    store.records_backend.merge("t-slow", _running_rec("t-slow"))

    current, outcome, reason = watch(app, poll=0.05, console=_quiet(), timeout=0.2)
    assert outcome == "timeout" and reason is not None and "still in flight" in reason
    assert RunState(current[0]["state"]) == RunState.RUNNING  # nothing was cancelled — read-only


def test_drive_stops_on_cancelled_instead_of_spinning(tmp_path: Path):
    """A CANCELLED task is terminal: ``drive_and_watch`` must stop (raise), not busy-loop ticking a DAG that can never progress (nothing RUNNING, no FAILED)."""
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
    """The wedge fix end-to-end: a worker killed *while watching* settles FAILED via ``reap_dead``, so the drain raises instead of waiting on it forever."""
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

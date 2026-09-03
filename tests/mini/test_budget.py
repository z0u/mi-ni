"""Wall-clock (cost) budget for detached runs (issue #14).

A budget stamps a ``deadline_at`` into the run's control plane at launch; any process that already polls the store enforces it opportunistically, so a forgotten or wedged detached run settles cleanly (CANCELLED) instead of burning money/resources indefinitely. These tests cover the metadata sidecar, the enforcement primitive, and the CLI wiring.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pytest

from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus
from mini.memo import META_KEY, MemoStore, PollCache
from mini.orchestration import tick
from mini.runs import RunState


def _slow_exp(name: str):
    def slow(x):
        import time

        time.sleep(30)  # only a cancel ends it within the test
        return x

    return Experiment(name=name, main=lambda ctx: ctx.map(slow, [1]))


def _reap(pid: int) -> None:
    """Wait for a SIGTERM'd worker to exit (confirms the kill + avoids a zombie)."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return
        except ChildProcessError:
            return  # already reaped, by the test body or an earlier sweep
        time.sleep(0.05)
    raise AssertionError("worker did not exit after cancel")


@pytest.fixture
def slow_run(tmp_path: Path):
    """Launch detached workers that only a cancel ends, and reap them all afterwards.

    Every budget test needs real in-flight work to tear down, so each one launches a ``sleep(30)`` subprocess; without a guaranteed teardown a failing assertion would leak it past the end of the session. Teardown cancels whatever is still in flight and waits for each pid, leaving the test bodies to say only what they are checking.
    """
    launched: list[tuple[LocalApparatus, MemoStore, dict]] = []

    def launch(
        name: str, app: LocalApparatus | None = None, exp: Experiment | None = None
    ) -> tuple[LocalApparatus, MemoStore, dict]:
        app = app if app is not None else LocalApparatus(name, data_dir=tmp_path / name)
        tick(exp if exp is not None else _slow_exp(name), app)  # launch + suspend
        store = app.memo_store()
        (rec,) = store.records()
        launched.append((app, store, rec))
        return app, store, rec

    yield launch

    for app, store, rec in launched:
        app.cancel(store)  # settled tasks are left alone, so this is a no-op if already torn down
        _reap(rec["pid"])


def test_meta_is_a_sidecar_excluded_from_records(tmp_path: Path):
    """Run-level metadata shares the record store but never reads as a task."""
    store = MemoStore(tmp_path / "exp")
    store.records_backend.write("train-abc123", {"key": "train-abc123", "state": RunState.DONE})
    store.set_meta(budget="30m", deadline_at=1234.0)

    assert store.deadline() == 1234.0
    assert store.meta()["budget"] == "30m"
    # The meta key is hidden from every records view, so it can't pollute status
    # output or skew the aggregate state.
    assert [r["key"] for r in store.records()] == ["train-abc123"]
    assert all(r["key"] != META_KEY for r in PollCache().records(store))


def test_budget_expired_gates_on_the_deadline(tmp_path: Path):
    store = MemoStore(tmp_path / "exp")
    assert store.budget_expired() is False  # unbudgeted → never expires

    store.set_meta(deadline_at=time.time() + 60)
    assert store.budget_expired() is False  # deadline in the future

    store.set_meta(deadline_at=time.time() - 1)
    assert store.budget_expired() is True  # deadline passed


def test_enforce_budget_tears_down_an_over_budget_run(slow_run):
    """A no-op while there is headroom; past the deadline, enforcement cancels in-flight work and really kills it."""
    app, store, rec = slow_run("budgetexp")
    assert RunState(rec["state"]) == RunState.RUNNING

    store.set_meta(deadline_at=time.time() + 60)  # plenty of headroom
    assert app.enforce_budget(store) == []  # nothing cancelled
    assert RunState(store.records()[0]["state"]) == RunState.RUNNING

    store.set_meta(budget="0s", deadline_at=time.time() - 1)  # already blown
    assert app.enforce_budget(store) == [rec["key"]]
    assert all(RunState(r["state"]) == RunState.CANCELLED for r in store.records())
    _reap(rec["pid"])  # the worker took the SIGTERM


def test_budget_is_scoped_per_experiment(slow_run):
    """Enforcing one experiment's budget must not touch a *different* experiment.

    Each experiment has its own control plane (a per-name dir locally, a ``mini-cp-<name>`` Dict on Modal), so the reserved ``META_KEY`` and the ``cancel`` that ``enforce_budget`` triggers are scoped to a single run — a concurrently-running, unbudgeted experiment is left strictly alone.
    """
    over, over_store, over_rec = slow_run("budget-over")  # both launch a long-running detached worker
    other, other_store, _ = slow_run("budget-other")
    over_store.set_meta(budget="0s", deadline_at=time.time() - 1)  # only this one is over budget

    cancelled = over.enforce_budget(over_store)

    assert cancelled == [over_rec["key"]]  # the budgeted run is torn down
    assert RunState(over_store.records()[0]["state"]) == RunState.CANCELLED
    # The other experiment is untouched: still RUNNING, and no budget leaked onto it.
    assert RunState(other_store.records()[0]["state"]) == RunState.RUNNING
    assert other_store.deadline() is None
    assert other.enforce_budget(other_store) == []  # unbudgeted → never tears down
    _reap(over_rec["pid"])  # the budgeted run's worker took the SIGTERM


def test_arm_budget_arms_then_inherits(tmp_path: Path):
    """``--budget`` (re)arms relative to now; a plain re-run inherits the deadline."""
    from mini.__main__ import _arm_budget

    store = MemoStore(tmp_path / "exp")

    _arm_budget(store, argparse.Namespace(budget="1h"))
    first = store.deadline()
    assert first is not None and abs(first - (time.time() + 3600)) < 5

    _arm_budget(store, argparse.Namespace(budget=None))  # no flag → unchanged
    assert store.deadline() == first

    time.sleep(0.01)
    _arm_budget(store, argparse.Namespace(budget="2h"))  # explicit flag re-arms
    assert store.deadline() != first
    assert abs(store.deadline() - (time.time() + 7200)) < 5  # ty:ignore[unsupported-operator]


def test_watch_driver_tears_down_over_budget_run(slow_run):
    """``run --watch`` stops at the deadline: it cancels in-flight work and raises ``BudgetExpired`` (an intentional teardown) rather than driving on."""
    import io

    from rich.console import Console

    from mini.monitor import drive_and_watch
    from mini.orchestration import BudgetExpired

    exp = _slow_exp("budgetwatch")
    app, store, rec = slow_run("budgetwatch", exp=exp)  # launches the detached worker (RUNNING)
    store.set_meta(budget="1m", deadline_at=time.time() - 1)  # already over budget

    with pytest.raises(BudgetExpired) as exc:
        drive_and_watch(exp, app, poll=0.05, console=Console(file=io.StringIO()))
    assert exc.value.cancelled == [rec["key"]]
    assert all(RunState(r["state"]) == RunState.CANCELLED for r in store.records())
    _reap(rec["pid"])


def test_status_enforces_budget_when_polled(tmp_path: Path, monkeypatch, capsys, slow_run):
    """A forgotten over-budget run settles CANCELLED the next time `status` reads it."""
    monkeypatch.chdir(tmp_path)  # no project marker → store resolves under cwd (.mini/<name>)
    from mini.__main__ import cmd_status

    # Default data_dir → .mini/budgetstatus, which is what `status` will resolve.
    _, store, rec = slow_run("budgetstatus", LocalApparatus("budgetstatus"))
    store.set_meta(budget="5m", deadline_at=time.time() - 1)  # expired

    cmd_status(argparse.Namespace(name="budgetstatus", app="local"))
    out = capsys.readouterr().out
    assert "CANCELLED" in out.upper()  # the run was torn down on poll
    assert "budget 5m" in out and "expired" in out  # and the tag is surfaced
    assert all(RunState(r["state"]) == RunState.CANCELLED for r in store.records())
    _reap(rec["pid"])


def _run_args(path: str) -> argparse.Namespace:
    return argparse.Namespace(
        path=path, watch=False, poll=0.05, app="local", workers=1, budget=None, keep_stale=False, key=None
    )


EXP_FILE = """
from mini.experiment import Experiment

def stage(x):
    return x

def main(ctx):
    first = ctx.map(stage, [1])          # suspends here on the first wake
    return ctx.run(stage, first[0] + 1)  # a second stage, only reached later

experiment = Experiment(name={name!r}, main=main)
"""


def test_run_past_an_expired_budget_says_how_to_re_arm(tmp_path: Path, monkeypatch, capsys):
    """The trap: past the deadline a plain ``run`` launches nothing at all, so a run with genuinely stale work reads exactly like one with nothing left to do — silence either way. It has to say that the budget is what's holding it, and hand back a command that gets it moving."""
    monkeypatch.chdir(tmp_path)
    from mini.__main__ import cmd_run

    path = tmp_path / "experiment.py"
    path.write_text(EXP_FILE.format(name="rearm"))
    LocalApparatus("rearm").memo_store().set_meta(budget="5m", deadline_at=time.time() - 1)

    cmd_run(_run_args(str(path)))
    out = capsys.readouterr().out
    assert "budget elapsed" in out
    assert "re-arm with" in out and "--budget 5m" in out  # a command that runs as printed
    assert not LocalApparatus("rearm").memo_store().records(), "nothing may launch past the deadline"


def _drain(app: LocalApparatus, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(RunState(r["state"]) == RunState.RUNNING for r in app.memo_store().records()):
            return
        time.sleep(0.05)
    raise AssertionError("tasks did not settle")


def test_status_distinguishes_a_finished_dag_from_a_suspended_one(tmp_path: Path, monkeypatch, capsys):
    """Every launched task DONE is not the same as the DAG being finished: a wake that suspended part-way leaves later stages un-launched, and the records read identically either way. A view that can't tick has to be told which it is, or a run sits looking complete with its publish step never executed."""
    monkeypatch.chdir(tmp_path)
    import json

    from mini.__main__ import cmd_ls, cmd_run, cmd_status

    path = tmp_path / "experiment.py"
    path.write_text(EXP_FILE.format(name="twostage"))
    app = LocalApparatus("twostage")

    cmd_run(_run_args(str(path)))  # launches stage one, suspends
    capsys.readouterr()
    _drain(app)

    cmd_status(argparse.Namespace(name="twostage", app="local", json=True, brief=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "done" and payload["settled"] is True  # …judging by the records alone
    assert payload["dag_complete"] is False  # …but there is more to run

    cmd_status(argparse.Namespace(name="twostage", app="local", json=False, brief=False))
    assert "DAG suspended" in capsys.readouterr().out
    cmd_ls(argparse.Namespace())
    assert "DAG suspended" in capsys.readouterr().out

    cmd_run(_run_args(str(path)))  # advance: launches stage two
    capsys.readouterr()
    _drain(app)
    cmd_run(_run_args(str(path)))  # the wake that runs main to the end
    capsys.readouterr()

    cmd_status(argparse.Namespace(name="twostage", app="local", json=True, brief=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["dag_complete"] is True
    cmd_status(argparse.Namespace(name="twostage", app="local", json=False, brief=False))
    assert "DAG suspended" not in capsys.readouterr().out

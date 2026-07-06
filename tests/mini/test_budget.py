"""Wall-clock (cost) budget for detached runs (issue #14).

A budget stamps a ``deadline_at`` into the run's control plane at launch; any
process that already polls the store enforces it opportunistically, so a
forgotten or wedged detached run settles cleanly (CANCELLED) instead of burning
money/resources indefinitely. These tests cover the metadata sidecar, the
enforcement primitive, and the CLI wiring.
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
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return
        time.sleep(0.05)
    raise AssertionError("worker did not exit after cancel")


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


def test_enforce_budget_noop_before_deadline(tmp_path: Path):
    app = LocalApparatus("budgetnoop", data_dir=tmp_path / "budgetnoop")
    tick(_slow_exp("budgetnoop"), app)  # launch + suspend
    store = app.memo_store()
    (rec,) = store.records()
    store.set_meta(deadline_at=time.time() + 60)  # plenty of headroom

    assert app.enforce_budget(store) == []  # nothing cancelled
    assert RunState(store.records()[0]["state"]) == RunState.RUNNING

    app.cancel(store)  # clean up the real worker
    _reap(rec["pid"])


def test_enforce_budget_tears_down_an_over_budget_run(tmp_path: Path):
    """Past the deadline, enforcement cancels in-flight work and really kills it."""
    app = LocalApparatus("budgetexp", data_dir=tmp_path / "budgetexp")
    tick(_slow_exp("budgetexp"), app)  # launch + suspend
    store = app.memo_store()
    (rec,) = store.records()
    pid = rec["pid"]
    assert RunState(rec["state"]) == RunState.RUNNING

    store.set_meta(budget="0s", deadline_at=time.time() - 1)  # already blown
    assert app.enforce_budget(store) == [rec["key"]]
    assert all(RunState(r["state"]) == RunState.CANCELLED for r in store.records())
    _reap(pid)  # the worker took the SIGTERM


def test_budget_is_scoped_per_experiment(tmp_path: Path):
    """Enforcing one experiment's budget must not touch a *different* experiment.

    Each experiment has its own control plane (a per-name dir locally, a
    ``mini-cp-<name>`` Dict on Modal), so the reserved ``META_KEY`` and the
    ``cancel`` that ``enforce_budget`` triggers are scoped to a single run — a
    concurrently-running, unbudgeted experiment is left strictly alone.
    """
    over = LocalApparatus("budget-over", data_dir=tmp_path / "budget-over")
    other = LocalApparatus("budget-other", data_dir=tmp_path / "budget-other")
    tick(_slow_exp("budget-over"), over)  # both launch a long-running detached worker
    tick(_slow_exp("budget-other"), other)
    over_store, other_store = over.memo_store(), other.memo_store()
    (over_rec,), (other_rec,) = over_store.records(), other_store.records()
    over_store.set_meta(budget="0s", deadline_at=time.time() - 1)  # only this one is over budget

    cancelled = over.enforce_budget(over_store)

    assert cancelled == [over_rec["key"]]  # the budgeted run is torn down
    assert RunState(over_store.records()[0]["state"]) == RunState.CANCELLED
    # The other experiment is untouched: still RUNNING, and no budget leaked onto it.
    assert RunState(other_store.records()[0]["state"]) == RunState.RUNNING
    assert other_store.deadline() is None
    assert other.enforce_budget(other_store) == []  # unbudgeted → never tears down

    _reap(over_rec["pid"])
    other.cancel(other_store)  # clean up the survivor
    _reap(other_rec["pid"])


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


def test_watch_driver_tears_down_over_budget_run(tmp_path: Path):
    """``run --watch`` stops at the deadline: it cancels in-flight work and raises
    ``BudgetExpired`` (an intentional teardown) rather than driving on."""
    import io

    from rich.console import Console

    from mini.monitor import drive_and_watch
    from mini.orchestration import BudgetExpired

    app = LocalApparatus("budgetwatch", data_dir=tmp_path / "budgetwatch")
    exp = _slow_exp("budgetwatch")
    tick(exp, app)  # launch the detached worker (RUNNING)
    store = app.memo_store()
    (rec,) = store.records()
    store.set_meta(budget="1m", deadline_at=time.time() - 1)  # already over budget

    with pytest.raises(BudgetExpired) as exc:
        drive_and_watch(exp, app, poll=0.05, console=Console(file=io.StringIO()))
    assert exc.value.cancelled == [rec["key"]]
    assert all(RunState(r["state"]) == RunState.CANCELLED for r in store.records())
    _reap(rec["pid"])


def test_status_enforces_budget_when_polled(tmp_path: Path, monkeypatch, capsys):
    """A forgotten over-budget run settles CANCELLED the next time `status` reads it."""
    monkeypatch.chdir(tmp_path)  # no project marker → store resolves under cwd (.mini/<name>)
    from mini.__main__ import cmd_status

    app = LocalApparatus("budgetstatus")  # default data_dir → .mini/budgetstatus
    tick(_slow_exp("budgetstatus"), app)  # launch + suspend
    store = app.memo_store()
    (rec,) = store.records()
    store.set_meta(budget="5m", deadline_at=time.time() - 1)  # expired

    cmd_status(argparse.Namespace(name="budgetstatus", app="local"))
    out = capsys.readouterr().out
    assert "CANCELLED" in out.upper()  # the run was torn down on poll
    assert "budget 5m" in out and "expired" in out  # and the tag is surfaced
    assert all(RunState(r["state"]) == RunState.CANCELLED for r in store.records())
    _reap(rec["pid"])

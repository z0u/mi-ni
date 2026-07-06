"""``mini gc``: reclaim memo storage no current read path can reach.

The invariants under test, in order of how much damage violating them does:

- **Current records are untouchable** — a DONE record is a future memo hit and
  a FAILED one is terminal state; gc must not convert either into a relaunch.
- **Superseded records are collectible only when the manifest is trustworthy**:
  the last tick ran the DAG to completion (``complete``) and nothing is
  unsettled. A suspended tick's manifest is complete only up to the suspension
  point, so "not requested" doesn't yet mean "never requested again".
- **Attempt files are judged by reachability**: readers resolve through the
  record's current ``gen``, so files under a replaced generation are garbage,
  while the legacy ``error.txt`` stays live as ``MemoStore.error``'s fallback.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mini.experiment import Experiment
from mini.gc import apply_gc, plan_gc
from mini.local_apparatus import LocalApparatus
from mini.orchestration import tick
from mini.runs import RunState


def _sweep(name: str, fn, xs: list) -> Experiment:
    return Experiment(name=name, main=lambda ctx: ctx.map(fn, xs))


def _drive(exp: Experiment, app: LocalApparatus, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, _ = tick(exp, app)
        if done:
            return
        time.sleep(0.1)
    raise AssertionError("orchestration did not complete")


def _with_superseded(name: str, monkeypatch, tmp_path: Path) -> LocalApparatus:
    """A completed run with one superseded record: sweep [1,2,3], then drop 3."""
    monkeypatch.chdir(tmp_path)

    def train(x):
        return x * 2

    app = LocalApparatus(name)
    _drive(_sweep(name, train, [1, 2, 3]), app)
    _drive(_sweep(name, train, [1, 2]), app)  # config 3 removed → its record superseded
    return app


def test_tick_stamps_manifest_completeness(tmp_path: Path, monkeypatch):
    """A suspended tick's manifest is incomplete; a completing tick's covers the DAG."""
    monkeypatch.chdir(tmp_path)

    def slow(x):
        time.sleep(0.3)
        return x

    exp = _sweep("stamp", slow, [1])
    app = LocalApparatus("stamp")
    done, _ = tick(exp, app)  # spawns, then suspends on the in-flight task
    assert not done
    assert app.memo_store().meta().get("complete") is False
    _drive(exp, app)
    assert app.memo_store().meta().get("complete") is True


def test_superseded_record_collected_with_its_dir_and_call(tmp_path: Path, monkeypatch):
    app = _with_superseded("gcx", monkeypatch, tmp_path)
    store = app.memo_store()
    recs = store.records()
    current, superseded = store.split_current(recs)
    [dead] = superseded
    dead_dir = store.result_dir(dead["key"])
    assert dead_dir.is_dir()

    plan = plan_gc(store, recs)
    assert {i.key for i in plan.by_kind("superseded")} == {dead["key"]}
    assert plan.size > 0
    apply_gc(store, plan)

    assert {r["key"] for r in store.records()} == {r["key"] for r in current}
    assert not dead_dir.exists()
    assert not store._call(dead["key"]).exists()
    assert sorted(store.result(r["key"]) for r in current) == [2, 4]  # kept hits still resolve


def test_superseded_kept_unless_manifest_trustworthy(tmp_path: Path, monkeypatch):
    app = _with_superseded("gates", monkeypatch, tmp_path)
    store = app.memo_store()
    current, superseded = store.split_current(store.records())
    [dead] = superseded

    # An incomplete manifest (a suspended tick) closes the gate entirely.
    store.set_meta(complete=False)
    plan = plan_gc(store)
    assert not plan.by_kind("superseded")
    assert any("completion" in reason for reason in plan.kept)

    # So does any unsettled current task.
    store.set_meta(complete=True)
    live = current[0]
    orig = store.record(live["key"])
    store.update(live["key"], state=RunState.RUNNING)
    plan = plan_gc(store)
    assert not plan.by_kind("superseded")
    assert any("unsettled" in reason for reason in plan.kept)
    store.update(live["key"], state=orig["state"])

    # A superseded record whose worker is still alive is skipped individually.
    store.update(dead["key"], state=RunState.RUNNING)
    plan = plan_gc(store)
    assert not plan.by_kind("superseded")
    assert any("cancel" in reason for reason in plan.kept)


def test_current_records_are_never_collected(tmp_path: Path, monkeypatch):
    """Even a FAILED current record is live state, not garbage."""
    monkeypatch.chdir(tmp_path)

    def bad(x):
        raise RuntimeError("bug")

    app = LocalApparatus("keepfail")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            tick(_sweep("keepfail", bad, [1]), app)
        except ExceptionGroup:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("map never surfaced the failure")

    store = app.memo_store()
    [rec] = store.records()
    assert rec["state"] == RunState.FAILED
    plan = plan_gc(store)
    assert not plan.by_kind("superseded") and not plan.by_kind("attempt-files")
    apply_gc(store, plan)  # collects at most the staged call
    [rec] = store.records()
    assert rec["state"] == RunState.FAILED  # still terminal, still visible
    assert "bug" in store.error(rec["key"])  # traceback intact


def test_stale_attempt_files_swept_current_and_unknown_kept(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def train(x):
        return x * 2

    app = LocalApparatus("attempts")
    _drive(_sweep("attempts", train, [1]), app)
    store = app.memo_store()
    [rec] = store.records()
    key, d = rec["key"], store.result_dir(rec["key"])
    assert rec["gen"]

    (d / "result-deadbeef.pkl").write_bytes(b"old attempt")  # replaced generation
    (d / "error-deadbeef.txt").write_text("old traceback")
    (d / "result.pkl").write_bytes(b"legacy result")  # unreachable: result() has no legacy fallback
    (d / "error.txt").write_text("legacy traceback")  # live: error() falls back to it
    (d / "notes.txt").write_text("unknown is not garbage")

    plan = plan_gc(store)
    [item] = plan.by_kind("attempt-files")
    assert set(item.names) == {"result-deadbeef.pkl", "error-deadbeef.txt", "result.pkl"}
    apply_gc(store, plan)

    assert store.result(key) == 2  # current attempt intact
    assert (d / "error.txt").exists() and (d / "notes.txt").exists()


def test_orphan_dirs_and_settled_calls_collected(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def train(x):
        return x

    app = LocalApparatus("orphans")
    _drive(_sweep("orphans", train, [1]), app)
    store = app.memo_store()
    [rec] = store.records()

    ghost = store.data_dir / "_memo" / "ghost-000000000000"
    ghost.mkdir(parents=True)
    (ghost / "result-cafecafe.pkl").write_bytes(b"x" * 64)

    plan = plan_gc(store)
    assert [i.key for i in plan.by_kind("orphan-dir")] == ["ghost-000000000000"]
    assert [i.key for i in plan.by_kind("staged-call")] == [rec["key"]]  # spawn input, task settled

    # A RUNNING task's staged call stays: its worker may still need respawn input.
    store.update(rec["key"], state=RunState.RUNNING)
    assert not plan_gc(store).by_kind("staged-call")
    store.update(rec["key"], state=RunState.DONE)

    apply_gc(store, plan)
    assert not ghost.exists()
    assert not store._call(rec["key"]).exists()
    assert store.result(rec["key"]) == 1


def test_cmd_gc_dry_run_then_apply(tmp_path: Path, monkeypatch, capsys):
    app = _with_superseded("gccli", monkeypatch, tmp_path)
    store = app.memo_store()

    from mini.__main__ import cmd_gc, cmd_status

    cmd_gc(argparse.Namespace(name="gccli", app="local", apply=False, store=False))
    out = capsys.readouterr().out
    assert "dry run" in out and "superseded" in out
    assert len(store.records()) == 3  # nothing deleted

    cmd_gc(argparse.Namespace(name="gccli", app="local", apply=True, store=False))
    assert "reclaimed" in capsys.readouterr().out
    assert len(store.records()) == 2

    cmd_status(argparse.Namespace(name="gccli", app="local"))  # the cleaned store still reads done
    assert "—  done  (2 tasks)" in capsys.readouterr().out

    cmd_gc(argparse.Namespace(name="gccli", app="local", apply=False, store=False))  # idempotent: nothing left
    assert "nothing to collect" in capsys.readouterr().out


def test_modal_record_store_delete(tmp_path: Path):
    from mini.modal_apparatus import ModalRecordStore

    store = ModalRecordStore({"k": {"key": "k", "state": "done"}})
    store.delete("k")
    store.delete("missing")  # absent key is a no-op, not an error
    assert store.keys() == []

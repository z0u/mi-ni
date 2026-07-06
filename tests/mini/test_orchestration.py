"""Tests for memoized multi-step orchestration (ctx.run/ctx.map + tick).

Task functions are *local* so cloudpickle serializes them by value; the
orchestration ``main`` runs in-process in the driver.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from mini.apparatus import Apparatus
from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus
from mini.memo import LocalRecordStore, MemoStore, task_key
from mini.orchestration import TaskFailed, retry, tick
from mini.runs import RunState


def _setup(name: str, main, tmp_path: Path) -> tuple[Experiment, LocalApparatus]:
    """An experiment (no compute) plus the apparatus it's run on (injected)."""
    return Experiment(name=name, main=main), LocalApparatus(name, data_dir=tmp_path / name)


def _drive(exp: Experiment, app: LocalApparatus, timeout: float = 30.0, keep_stale: bool = False):
    """Re-run the orchestration each 'wake' until it completes (mirrors the agent loop)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, payload = tick(exp, app, keep_stale=keep_stale)
        if done:
            return payload
        time.sleep(0.1)
    raise AssertionError(f"orchestration did not complete: {payload}")


def _drive_to_failure(exp: Experiment, app: LocalApparatus, timeout: float = 30.0) -> ExceptionGroup:
    """Tick until the strict map surfaces its failures; return the group."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            tick(exp, app)
        except ExceptionGroup as eg:
            return eg
        time.sleep(0.1)
    raise AssertionError("map never surfaced the failure")


def test_multistep_dependency(tmp_path: Path):
    def prep():
        return {"vocab": 7}

    def train(lr, vocab):
        return {"lr": lr, "vocab": vocab}

    def main(ctx):
        meta = ctx.run(prep)  # sweep configs depend on prep's output
        return ctx.map(train, [0.1, 0.2], [meta["vocab"]] * 2)

    assert _drive(*_setup("dep", main, tmp_path)) == [{"lr": 0.1, "vocab": 7}, {"lr": 0.2, "vocab": 7}]


def test_prep_runs_once_across_wakes(tmp_path: Path):
    def prep():
        from mini import get_data_dir

        f = get_data_dir() / "prep_count"
        n = int(f.read_text()) if f.exists() else 0
        f.write_text(str(n + 1))
        return n + 1

    def train(x):
        return x * 2

    def main(ctx):
        ctx.run(prep)
        return ctx.map(train, [1, 2])

    exp, app = _setup("once", main, tmp_path)
    _drive(exp, app)
    tick(exp, app)  # extra wakes after completion
    tick(exp, app)
    assert (tmp_path / "once" / "prep_count").read_text() == "1"  # prep memoized, ran once


def test_failed_is_terminal_until_retry(tmp_path: Path):
    """A thrown task settles FAILED and does *not* auto-relaunch: the strict map
    keeps raising on every wake. An explicit ``retry`` resets it so the next drive
    reruns just that task and completes."""

    def train(x):
        from mini import get_data_dir

        f = get_data_dir() / f"att_{x}"
        n = int(f.read_text()) if f.exists() else 0
        f.write_text(str(n + 1))
        if x == 2 and n == 0:  # fails on the first attempt only
            raise RuntimeError("transient")
        return x * 10

    def main(ctx):
        return ctx.map(train, [1, 2, 3])

    exp, app = _setup("crash", main, tmp_path)
    store = app.memo_store()

    # Drive until the map surfaces the failure (task 2 throws; 1 & 3 succeed).
    _drive_to_failure(exp, app)
    # FAILED is terminal (the code hasn't changed): re-ticking keeps raising, never relaunches.
    for _ in range(3):
        with pytest.raises(ExceptionGroup) as exc:
            tick(exp, app)
        assert all(isinstance(e, TaskFailed) for e in exc.value.exceptions)
        time.sleep(0.1)
    states = {r["key"]: r.get("state") for r in store.records()}
    assert sum(s == RunState.FAILED for s in states.values()) == 1  # not relaunched
    assert (tmp_path / "crash" / "att_2").read_text() == "1"  # threw exactly once

    # Explicit retry heals: reset the failed task, then drive to completion.
    assert len(retry(store)) == 1  # one FAILED task reset
    assert _drive(exp, app) == [10, 20, 30]
    assert (tmp_path / "crash" / "att_2").read_text() == "2"  # ran again on retry
    assert (tmp_path / "crash" / "att_1").read_text() == "1"  # siblings untouched


def test_allow_partial_returns_sentinel_for_failed_cell(tmp_path: Path):
    """``allow_partial=True`` lets a map settle with a failed cell instead of
    blocking forever: the result stays index-aligned with the inputs, with
    ``MISSING`` where a task failed, so a downstream reduce can run on the rest."""
    from mini.orchestration import MISSING

    def train(x):
        if x == 2:
            raise RuntimeError("bad region")
        return x * 10

    def main(ctx):
        results = ctx.map(train, [1, 2, 3], allow_partial=True)
        present = [r for r in results if r is not MISSING]
        return {"results": results, "best": max(present)}

    out = _drive(*_setup("partial", main, tmp_path))
    assert out["results"] == [10, MISSING, 30]  # index-aligned; gap where x=2 failed
    assert out["best"] == 30  # downstream computation ran on the surviving subset


def test_allow_partial_still_waits_for_in_flight(tmp_path: Path):
    """Partial is not best-effort: it waits for running tasks to settle before
    returning, so a slow-but-fine cell still lands a real result (not a gap)."""

    def train(x):
        if x == 2:
            raise RuntimeError("nope")
        time.sleep(0.3)  # outlives the first wake; must be awaited, not skipped
        return x * 10

    def main(ctx):
        return ctx.map(train, [1, 2, 3], allow_partial=True)

    from mini.orchestration import MISSING

    assert _drive(*_setup("partwait", main, tmp_path)) == [10, MISSING, 30]


def test_strict_map_surfaces_failures_as_group(tmp_path: Path):
    """Without ``allow_partial`` failed cells are terminal: once the fan-out has
    settled, the map raises an ``ExceptionGroup`` of ``TaskFailed`` — *all* the
    failures at once, after waiting for the slower siblings to settle too."""

    def train(x):
        if x in (2, 3):
            raise RuntimeError("boom")
        return x

    exp, app = _setup("strict", lambda ctx: ctx.map(train, [1, 2, 3]), tmp_path)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            done, _ = tick(exp, app)
        except ExceptionGroup as eg:
            assert len(eg.exceptions) == 2  # both failing cells, surfaced together
            failures = [e for e in eg.exceptions if isinstance(e, TaskFailed)]
            assert len(failures) == 2  # all of them are TaskFailed
            assert {e.state for e in failures} == {RunState.FAILED}
            # the original exception's type rides along as a string, so a caller can
            # bucket failures by kind without importing the worker's libraries
            assert {e.exc_type for e in failures} == {"builtins.RuntimeError"}
            return
        assert not done, "strict map completed despite a failed cell"
        time.sleep(0.1)
    raise AssertionError("strict map never surfaced its failures")


def test_missing_sentinel_is_falsey_singleton_and_pickles(tmp_path: Path):
    """``MISSING`` is a falsey singleton distinct from ``None``, and survives a
    (cloud)pickle round-trip as the *same* object — so ``r is MISSING`` holds in a
    downstream task that receives a partial result over the wire."""
    import pickle

    from mini import MISSING

    assert not MISSING and MISSING is not None
    assert bool(MISSING) is False
    assert pickle.loads(pickle.dumps(MISSING)) is MISSING
    assert repr(MISSING) == "<missing>"


def test_single_map(tmp_path: Path):
    def sq(x):
        return x * x

    exp = Experiment(name="map", main=lambda ctx: ctx.map(sq, [2, 3]))
    app = LocalApparatus("map", data_dir=tmp_path / "map")
    assert _drive(exp, app) == [4, 9]


def test_map_does_not_unpack_tuple_items(tmp_path: Path):
    """A single-iterable map passes each element as *one* argument — an element
    that happens to be a tuple stays a tuple. (The old items-based map unpacked
    tuples as positional args, silently breaking tasks that take a tuple.)"""

    def span(pair):
        lo, hi = pair
        return hi - lo

    exp, app = _setup("tup", lambda ctx: ctx.map(span, [(1, 4), (2, 8)]), tmp_path)
    assert _drive(exp, app) == [3, 6]


def test_map_zips_iterables_strictly(tmp_path: Path):
    """Multiple iterables zip Executor-style into positional args — and
    mismatched lengths raise rather than silently truncating the sweep."""

    def add(a, b):
        return a + b

    exp, app = _setup("zipped", lambda ctx: ctx.map(add, [1, 2], [10, 20]), tmp_path)
    assert _drive(exp, app) == [11, 22]

    exp, app = _setup("lopsided", lambda ctx: ctx.map(add, [1, 2], [10]), tmp_path)
    with pytest.raises(ValueError, match="shorter"):
        tick(exp, app)


def test_metrics_recorded_on_task(tmp_path: Path):
    def t(x):
        from mini import emit_metrics

        emit_metrics(v=float(x))
        return x

    def main(ctx):
        return ctx.map(t, [5])

    _drive(*_setup("met", main, tmp_path))
    recs = MemoStore(tmp_path / "met").records()
    assert any(r.get("metrics", {}).get("v") == 5.0 for r in recs)


def test_env_recorded_on_task(tmp_path: Path):
    """The worker stamps each record with *what it ran on* (host/OS/Python)."""
    _drive(*_setup("env", lambda ctx: ctx.map(lambda x: x, [5]), tmp_path))
    (rec,) = MemoStore(tmp_path / "env").records()
    env = rec["env"]
    assert env["host"] and env["platform"] and env["python"]


class _CountingStore(LocalRecordStore):
    """A ``LocalRecordStore`` that records which keys were read, to assert the
    cache stops re-reading the settled tail."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.reads: list[str] = []

    def read(self, key: str) -> dict[str, Any] | None:
        self.reads.append(key)
        return super().read(key)


def test_poll_cache_reads_settled_records_once(tmp_path: Path):
    """``PollCache`` serves the immutable settled tail from memory: a key is read
    from the backend exactly once after it settles, then never again."""
    from mini.memo import PollCache

    backend = _CountingStore(tmp_path / "pc")
    store = MemoStore(tmp_path / "pc", records=backend)
    backend.write("a", {"key": "a", "state": RunState.RUNNING})
    backend.write("b", {"key": "b", "state": RunState.DONE})

    cache = PollCache()
    cache.records(store)  # a: RUNNING (re-read), b: DONE (cached)
    cache.records(store)  # a re-read again; b served from cache
    assert backend.reads.count("b") == 1 and backend.reads.count("a") == 2

    backend.write("a", {"key": "a", "state": RunState.FAILED})  # a settles
    backend.reads.clear()
    states = {r["key"]: r["state"] for r in cache.records(store)}  # picks up FAILED, caches it
    cache.records(store)
    assert states["a"] == RunState.FAILED
    assert backend.reads.count("a") == 1  # read once on settle, then cached — no re-read


def test_version_reruns_in_place(tmp_path: Path):
    """``version=`` is evidence, not identity: a bump re-runs the task as a new
    attempt on the *same* record, with the old attempt kept in its history."""

    def t(x):
        return x

    def main_v1(ctx):
        return ctx.map(t, [1], version="v1")

    def main_v2(ctx):
        return ctx.map(t, [1], version="v2")

    _drive(*_setup("ver", main_v1, tmp_path))
    _drive(Experiment(name="ver", main=main_v2), LocalApparatus("ver", data_dir=tmp_path / "ver"))
    (rec,) = MemoStore(tmp_path / "ver").records()  # one identity across both versions
    assert rec["version"] == "v2" and rec["state"] == RunState.DONE
    assert [a["version"] for a in rec["history"]] == ["v1"]  # the bump preserved the story


def test_prune_and_memo_hits_across_config_edits(tmp_path: Path):
    """Editing a sweep's config set re-runs only what changed.

    The fix/prune/retry contract for a `ctx.map`: re-running with a different set
    of items leaves unchanged items as memo hits (not relaunched), runs only the
    new/changed items, and simply stops requesting a removed item. Proven with a
    per-arg execution counter on the volume, so a memo hit shows count == 1."""
    counts = tmp_path / "counts"
    counts.mkdir()

    def work(x):
        marker = counts / str(x)  # side effect: how many times this arg actually ran
        marker.write_text(str(int(marker.read_text()) + 1 if marker.exists() else 1))
        return x * 10

    def sweep(items):
        return Experiment(name="prune", main=lambda ctx: ctx.map(work, items))

    data_dir = tmp_path / "prune"  # one shared memo store across both drives
    assert _drive(sweep([1, 2]), LocalApparatus("prune", data_dir=data_dir, max_workers=3)) == [10, 20]
    assert {p.name for p in counts.iterdir()} == {"1", "2"}

    # Drop 1, keep 2, add 3: only 3 runs; 2 is a memo hit; 1 is no longer requested.
    assert _drive(sweep([2, 3]), LocalApparatus("prune", data_dir=data_dir, max_workers=3)) == [20, 30]
    assert (counts / "2").read_text() == "1", "unchanged item re-ran instead of hitting the memo"
    assert (counts / "3").read_text() == "1", "added item did not run exactly once"
    assert {p.name for p in counts.iterdir()} == {"1", "2", "3"}  # 1 retained, never re-run


def test_tick_persists_requested_keys(tmp_path: Path):
    """Each tick records the keys the DAG requested (the ``__run__`` manifest), so
    read-only views can split current records from superseded ones without
    re-running ``main`` (reads must never tick)."""

    def work(x):
        return x * 10

    def sweep(items):
        return Experiment(name="req", main=lambda ctx: ctx.map(work, items))

    data_dir = tmp_path / "req"
    _drive(sweep([1, 2]), LocalApparatus("req", data_dir=data_dir))
    store = MemoStore(data_dir)
    assert set(store.requested_keys() or []) == {r["key"] for r in store.records()}

    # Prune a config: its record survives on disk, but the manifest no longer
    # requests it — so the run's *current* view is just the surviving cell.
    _drive(sweep([2]), LocalApparatus("req", data_dir=data_dir))
    current, stale = store.split_current(store.records())
    assert len(current) == 1 and len(stale) == 1


def test_superseded_records_are_excluded_and_not_retried(tmp_path: Path):
    """*Renaming* (replacing) a task fn changes its identity, orphaning the old
    records — an in-place edit does not (see the hotfix tests). The orphaned
    FAILED record must not poison the run: ``retry`` skips it (resetting it would
    plant a phantom no tick ever relaunches), and the manifest marks it
    superseded for read-only views. Explicit ``--key`` intent still beats the
    manifest."""

    def bad(x):
        if x == 2:
            raise RuntimeError("bug")
        return x * 10

    def good(x):
        return x * 10

    def sweep(fn):
        return Experiment(name="hotfix", main=lambda ctx: ctx.map(fn, [1, 2]))

    data_dir = tmp_path / "hotfix"
    _drive_to_failure(sweep(bad), LocalApparatus("hotfix", data_dir=data_dir))
    store = LocalApparatus("hotfix", data_dir=data_dir).memo_store()
    (failed_key,) = [r["key"] for r in store.records() if r.get("state") == RunState.FAILED]

    # The replacement fn is a new identity (different name), so every cell re-keys
    # and the old records are superseded. The new sweep completes despite the old
    # failure.
    assert _drive(sweep(good), LocalApparatus("hotfix", data_dir=data_dir)) == [10, 20]
    assert retry(store) == []  # the orphaned FAILED is skipped, not reset
    assert store.state(failed_key) == RunState.FAILED  # left as-is — no phantom pending

    current, stale = store.split_current(store.records())
    assert failed_key in {r["key"] for r in stale}
    assert all(r.get("state") == RunState.DONE for r in current)

    assert retry(store, key=failed_key) == [failed_key]  # explicit key overrides


def _make_train(fixed: bool):
    """Two variants of the *same* task fn — same module and qualname, different
    source: an in-place edit, as far as identity is concerned."""
    if fixed:

        def train(x):
            from mini import get_data_dir

            f = get_data_dir() / f"ran_{x}"
            f.write_text(str(int(f.read_text()) + 1 if f.exists() else 1))
            return x * 10
    else:

        def train(x):
            from mini import get_data_dir

            f = get_data_dir() / f"ran_{x}"
            f.write_text(str(int(f.read_text()) + 1 if f.exists() else 1))
            if x == 2:
                raise RuntimeError("bug")
            return x * 10

    return train


def _hotfix_sweep(fixed: bool) -> Experiment:
    return Experiment(name="hot", main=lambda ctx: ctx.map(_make_train(fixed), [1, 2]))


def test_hotfix_edit_relaunches_failed_cells_in_place(tmp_path: Path):
    """The sweep-hotfix story. Editing the fn moves its *evidence*, not its keys:
    the FAILED cell relaunches automatically (the fix is what it was waiting for
    — no ``retry``), nothing is orphaned, and by default the stale DONE cell
    re-runs too (bias to over-invalidate). The healed record keeps the failed
    attempt in its history."""
    data_dir = tmp_path / "hot"
    _drive_to_failure(_hotfix_sweep(False), LocalApparatus("hot", data_dir=data_dir))
    store = LocalApparatus("hot", data_dir=data_dir).memo_store()
    keys_before = {r["key"] for r in store.records()}

    assert _drive(_hotfix_sweep(True), LocalApparatus("hot", data_dir=data_dir)) == [10, 20]
    assert {r["key"] for r in store.records()} == keys_before  # same identities — nothing orphaned
    assert store.split_current(store.records())[1] == []  # no superseded records
    assert {x: int((data_dir / f"ran_{x}").read_text()) for x in (1, 2)} == {1: 2, 2: 2}

    healed = store.record(task_key(_make_train(True), (2,)))
    assert healed["state"] == RunState.DONE  # healed in place, same address
    (prior,) = healed["history"]
    assert prior["state"] == RunState.FAILED and prior["code_fp"] != healed["code_fp"]  # the edit is why it re-ran


def test_keep_stale_bounds_hotfix_to_unfinished_cells(tmp_path: Path):
    """``--keep-stale-done``: after an edit, DONE cells are served as-is and only
    the cells that never finished re-run with the new code — the bounded hotfix.
    The kept key lands in run meta so read-only views can badge it."""
    data_dir = tmp_path / "hot"
    _drive_to_failure(_hotfix_sweep(False), LocalApparatus("hot", data_dir=data_dir))
    store = LocalApparatus("hot", data_dir=data_dir).memo_store()

    assert _drive(_hotfix_sweep(True), LocalApparatus("hot", data_dir=data_dir), keep_stale=True) == [10, 20]
    assert {x: int((data_dir / f"ran_{x}").read_text()) for x in (1, 2)} == {1: 1, 2: 2}  # DONE cell untouched
    assert store.meta()["kept_stale"] == [task_key(_make_train(True), (1,))]


def test_per_step_apparatus_uses_its_hooks(tmp_path: Path):
    """``on=`` routes a step to a different apparatus — here proven via its hooks."""

    def mark_default():
        from mini import get_data_dir

        (get_data_dir() / "default_hook").touch()

    def mark_gpu():
        from mini import get_data_dir

        (get_data_dir() / "gpu_hook").touch()

    def task(x):
        return x

    data_dir = tmp_path / "perstep"
    default = LocalApparatus("perstep", data_dir=data_dir).before_each(mark_default)
    gpu = LocalApparatus("perstep", data_dir=data_dir).before_each(mark_gpu)

    def main(ctx):
        return ctx.map(task, [1], on=gpu)

    assert _drive(Experiment(name="perstep", main=main), default) == [1]
    assert (data_dir / "gpu_hook").exists()  # the on= apparatus's hook ran
    assert not (data_dir / "default_hook").exists()  # the tick default's did not


def test_role_routes_to_its_apparatus(tmp_path: Path):
    """``role=`` binds a label to a ``.w()`` variant via the experiment's ``roles``
    table — proven (like ``on=``) through each variant's ``before_each`` hook."""

    def mark_prep():
        from mini import get_data_dir

        (get_data_dir() / "prep_hook").touch()

    def mark_train():
        from mini import get_data_dir

        (get_data_dir() / "train_hook").touch()

    def task(x):
        return x

    data_dir = tmp_path / "roles"

    def roles(base: Apparatus) -> dict[str, Apparatus]:
        # callable form: lets each role attach its own hook (local has no .w knobs).
        # Typed against the base Apparatus so it matches Experiment.roles' contract
        # (the field's callable must accept any apparatus --app built, not just local).
        return {"prep": base.before_each(mark_prep), "train": base.before_each(mark_train)}

    def main(ctx):
        ctx.run(task, 0, role="prep")
        return ctx.map(task, [1], role="train")

    exp = Experiment(name="roles", main=main, roles=roles)
    assert _drive(exp, LocalApparatus("roles", data_dir=data_dir)) == [1]
    assert (data_dir / "prep_hook").exists() and (data_dir / "train_hook").exists()


def test_role_kwargs_table_applies_w(tmp_path: Path):
    """The dict form maps a label to ``.w()`` kwargs; the base apparatus's ``.w``
    interprets them (local ignores GPU knobs, so the same table runs locally)."""
    captured: dict[str, Any] = {}

    class RecordingLocal(LocalApparatus):
        def w(self, **kwargs):  # local has no native knobs; record + no-op
            captured.update(kwargs)
            return self

    def task(x):
        return x

    exp = Experiment(name="wtab", main=lambda ctx: ctx.map(task, [1], role="train"), roles={"train": dict(gpu="L4")})
    assert _drive(exp, RecordingLocal("wtab", data_dir=tmp_path / "wtab")) == [1]
    assert captured == {"gpu": "L4"}


def test_unknown_role_and_role_on_conflict_raise(tmp_path: Path):
    from mini.orchestration import Ctx

    app = LocalApparatus("routing", data_dir=tmp_path / "routing")
    ctx = Ctx(app.memo_store(), app, roles={"train": app})

    import pytest

    with pytest.raises(ValueError, match="unknown role"):
        ctx.run(lambda: None, role="gpu")
    with pytest.raises(ValueError, match="not both"):
        ctx.run(lambda: None, on=app, role="train")


def test_ctx_spawns_via_the_apparatus(tmp_path: Path):
    """``ctx`` launches tasks through ``apparatus.spawn_tasks`` — the seam the
    Modal backend implements — and batches a map's fan-out into one call. Proven
    by routing to an apparatus that runs tasks *synchronously in-process* instead
    of spawning: if Ctx bypassed the seam, the drive would time out."""
    from mini._taskworker import run_task
    from mini.local_apparatus import LocalApparatus

    batches: list[int] = []

    class InlineApparatus(LocalApparatus):
        def spawn_tasks(self, store, batch):
            batches.append(len(batch))  # record fan-out width
            for key, gen, fn, args, hooks in batch:
                store.write_call(key, fn, args, hooks, gen)
                run_task(store.data_dir, key)  # run now, in-process — no subprocess

    def task(x):
        return x * 3

    app = InlineApparatus("inline", data_dir=tmp_path / "inline")
    exp = Experiment(name="inline", main=lambda ctx: ctx.map(task, [2, 5]))
    assert _drive(exp, app) == [6, 15]
    assert batches == [2]  # both tasks launched in a single batched spawn


def test_task_key_is_deterministic_and_input_sensitive():
    def fn(x):
        return x

    assert task_key(fn, (1,)) == task_key(fn, (1,))
    assert task_key(fn, (1,)) != task_key(fn, (2,))


def test_input_fingerprint_stable_across_processes():
    """Inputs containing a set (e.g. a Pydantic model's ``__pydantic_fields_set__``)
    must fingerprint identically across processes — every agent wake is a fresh one,
    and ``PYTHONHASHSEED`` randomizes set order. A plain ``pickle.dumps`` would differ.
    """
    import os
    import subprocess
    import sys

    code = "from mini.memo import _input_fingerprint; print(_input_fingerprint(({'e', 'a', 'd', 'b', 'c'},)))"
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
        ).stdout.strip()
        for seed in ("0", "1", "2")
    }
    assert len(outs) == 1, f"fingerprint varied across hash seeds: {outs}"

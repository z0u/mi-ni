"""Concurrency safety of the memo control plane.

Two writers can share a record: a second ticker racing the first to launch, a
reaper racing a worker's final write, or a stale worker (superseded relaunch, or
cancelled but surviving SIGTERM) racing its successor. The generation stamp on
attempts plus the locked local record store keep every such race from corrupting
the record, the result, or a mutable name in the artifact store.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mini._taskworker import execute_task
from mini.local_apparatus import LocalApparatus
from mini.memo import LocalRecordStore, MemoStore, task_key_parts
from mini.runs import RunState
from mini.store import LocalStore


def _claim(store: MemoStore, fn, args=(1,)) -> tuple[str, str]:
    """Claim a fresh attempt as ``Ctx._classify`` would; return (key, gen)."""
    key, parts = task_key_parts(fn, args)
    gen = store.mark_running(fn, key, parts, expect_gen=store.record(key).get("gen"))
    assert gen is not None
    return key, gen


def _work(x):
    return x * 10


def test_mark_running_claim_is_exclusive(tmp_path: Path):
    """Two tickers that both read "un-run" cannot both claim the launch: the
    second write-if fails and returns None, so only one worker is ever spawned
    (the double-spawn race between the record read and mark_running)."""
    store = MemoStore(tmp_path / "claim")
    key, parts = task_key_parts(_work, (1,))
    # Both tickers read the record first (both see no attempt), *then* both claim.
    assert store.mark_running(_work, key, parts, expect_gen=None) is not None
    assert store.mark_running(_work, key, parts, expect_gen=None) is None  # lost the race — don't spawn


def test_stale_worker_is_fenced_after_supersession(tmp_path: Path):
    """Once a successor claims the record, the old attempt's writes stop landing:
    no heartbeat resurrection, no DONE over the new attempt's RUNNING."""
    store = MemoStore(tmp_path / "fence")
    key, old = _claim(store, _work)
    _, new = _claim(store, _work)  # relaunch (e.g. after a cancel): new generation

    assert store.update_if(key, old, state=RunState.DONE, heartbeat_at=1.0) is False
    assert store.record(key)["state"] == RunState.RUNNING  # untouched
    assert store.update_if(key, new, state=RunState.DONE) is True


def test_superseded_worker_exits_at_startup_without_writing(tmp_path: Path):
    """A worker whose attempt was superseded before it started runs nothing."""
    ran = tmp_path / "ran"
    store = MemoStore(tmp_path / "startup")

    def task(x):
        ran.touch()
        return x

    key, old = _claim(store, task)
    _claim(store, task)  # successor claims before the old worker starts
    execute_task(store, key, task, (1,), [], gen=old)
    assert not ran.exists()
    assert store.record(key)["state"] == RunState.RUNNING  # the successor's claim, untouched


def test_zombie_finishing_mid_run_cannot_overwrite_successor(tmp_path: Path):
    """The todo.md hazard: a worker superseded *mid-run* completes anyway. Its
    final state is fenced out and its result lands in its own generation's file,
    so the successor's record and result both survive."""
    store = MemoStore(tmp_path / "zombie")
    taken: dict[str, str] = {}

    def sneaky(x):
        # While the first attempt runs, a successor claims the record.
        _, taken["gen"] = _claim(store, sneaky)
        return "stale"

    key, old = _claim(store, sneaky)
    execute_task(store, key, sneaky, (1,), [], gen=old)  # zombie runs to completion
    assert store.record(key)["state"] == RunState.RUNNING  # DONE was fenced out

    # The successor completes normally and its result is the one served.
    execute_task(store, key, lambda x: "fresh", (1,), [], gen=taken["gen"])
    assert store.record(key)["state"] == RunState.DONE
    assert store.result(key) == "fresh"


def test_cancel_releases_the_generation(tmp_path: Path):
    """``cancel`` releases the record's gen, so a worker that survives SIGTERM
    can't flip CANCELLED back to DONE and pass its half-cancelled result off as
    current."""
    store = MemoStore(tmp_path / "cancel")
    key, gen = _claim(store, _work)
    app = LocalApparatus("cancel", data_dir=tmp_path / "cancel")
    assert app.cancel(store) == [key]

    assert store.update_if(key, gen, state=RunState.DONE) is False  # the survivor is fenced
    assert store.record(key)["state"] == RunState.CANCELLED


def test_stale_worker_cannot_move_a_ref(tmp_path: Path):
    """A worker superseded mid-run that tries ``set_ref`` fails loudly with
    ``StaleWriteError`` — the name never moves, and the successor's write is the
    one that resolves (issue #46: refs were the last unfenced write path)."""
    from mini.store import put, set_ref

    store = MemoStore(tmp_path / "refs")
    artifacts = LocalStore(tmp_path / "store")
    taken: dict[str, str] = {}

    def sneaky(x):
        _, taken["gen"] = _claim(store, sneaky)  # superseded mid-run
        set_ref("best", put(b"stale", name="model.bin"))

    key, old = _claim(store, sneaky)
    execute_task(store, key, sneaky, (1,), [], artifacts=artifacts, gen=old)
    assert artifacts.get_ref("best") is None  # the name never moved
    assert store.record(key)["state"] == RunState.RUNNING  # FAILED fenced out too
    assert "StaleWriteError" in store.error_path(key, old).read_text()

    def fresh(x):
        set_ref("best", put(b"fresh", name="model.bin"))
        return "ok"

    execute_task(store, key, fresh, (1,), [], artifacts=artifacts, gen=taken["gen"])
    art = artifacts.get_ref("best")
    assert art is not None
    assert artifacts.get(art, tmp_path / "out.bin").read_bytes() == b"fresh"


def test_stale_worker_cannot_publish(tmp_path: Path):
    """``publish`` is fenced like ``set_ref``: a superseded attempt cannot
    last-writer-win a published path."""
    from mini.store import publish, put

    store = MemoStore(tmp_path / "pub")
    artifacts = LocalStore(tmp_path / "store")

    def sneaky(x):
        _claim(store, sneaky)  # superseded mid-run
        publish(put(b"stale", name="fig.png"), "reports/fig.png")

    key, old = _claim(store, sneaky)
    execute_task(store, key, sneaky, (1,), [], artifacts=artifacts, gen=old)
    assert not (tmp_path / "store" / "published" / "reports" / "fig.png").exists()
    assert "StaleWriteError" in store.error_path(key, old).read_text()


def test_current_worker_name_writes_land(tmp_path: Path):
    """The fence has no false positives: the attempt that owns the record can
    ``set_ref`` and ``publish`` normally."""
    from mini.store import publish, put, set_ref

    store = MemoStore(tmp_path / "ok")
    artifacts = LocalStore(tmp_path / "store")

    def task(x):
        art = put(b"good", name="model.bin")
        set_ref("best", art)
        return publish(art, "reports/model.bin")

    key, gen = _claim(store, task)
    execute_task(store, key, task, (1,), [], artifacts=artifacts, gen=gen)
    assert store.record(key)["state"] == RunState.DONE
    assert artifacts.get_ref("best") is not None
    assert store.result(key).startswith("file://")  # publish's URL came back as the result
    assert (tmp_path / "store" / "published" / "reports" / "model.bin").read_bytes() == b"good"


def test_concurrent_merges_do_not_drop_fields(tmp_path: Path):
    """``merge`` is read-modify-write; unlocked, two mergers (heartbeat vs reaper)
    each read the same base and silently drop the other's fields. The store-wide
    lock serializes them: after a thrash of concurrent single-field merges, every
    field is present with its final value."""
    backend = LocalRecordStore(tmp_path / "merge")
    backend.write("k", {"key": "k"})

    def hammer(i: int) -> None:
        for n in range(25):
            backend.merge("k", {f"f{i}": n})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(hammer, range(8)))

    rec = backend.read("k")
    assert rec == {"key": "k", **{f"f{i}": 24 for i in range(8)}}

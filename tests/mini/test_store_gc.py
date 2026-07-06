"""``mini gc --store``: mark-and-sweep the content-addressed artifact CAS.

The sweep's contract, in order of how much damage getting it wrong does:

- **Fail closed.** ``collect_store_roots`` would rather refuse than under-mark:
  an in-flight task, an unreadable result, or an unknown backend stamp aborts
  with nothing deleted. Deleting a blob a running worker is about to reference
  (it just saw ``has() == True``) would corrupt a result not yet written.
- **Every record is a root** — current *and* superseded. Collecting a
  superseded record's blob is ``mini gc <name>``'s call to make first; the store
  sweep never second-guesses the memo layer, so a blob only becomes collectible
  once the record referencing it is gone.
- **Refs pin.** ``set_ref`` is the documented way to keep an artifact alive with
  no record referencing it, so a ref-pinned blob survives the sweep.
- **The grace window** keeps young blobs — the margin against writers the mark
  phase cannot see (an unpushed checkout, a ``put`` that skipped its upload just
  before the sweep judged the bytes garbage).

The forward artifact index (the ``result-<gen>.artifacts.json`` sidecar) is what
lets the mark phase read tiny JSON instead of unpickling every result; unpickling
stays the fallback for pre-sidecar records.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import pytest

from mini.experiment import Experiment
from mini.gc import (
    ModalGcIO,
    StoreGcError,
    apply_gc,
    apply_store_gc,
    collect_store_roots,
    plan_gc,
    plan_store_gc,
)
from mini.local_apparatus import LocalApparatus
from mini.memo import MemoStore
from mini.orchestration import tick
from mini.runs import data_root
from mini.store import Artifact, LocalStore, _cas_key, artifact_shas, store_for


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


def _put_step():
    """A step that stores a per-input blob and returns its handle (distinct bytes → distinct sha).

    Built as a local closure so cloudpickle serializes it *by value* — a
    module-level function would pickle by reference, and the detached subprocess
    worker can't import this test module.
    """

    def put_step(x):
        from mini.store import put

        return put(f"artifact-{x}".encode(), name=f"data-{x}.bin")

    return put_step


def _shas(plan) -> set[str]:
    return {b.sha256 for b in plan.unreferenced}


# ---------------------------------------------------------------------------
# artifact_shas: the object-graph walk the mark phase reads from
# ---------------------------------------------------------------------------


def _file(sha: str, name: str = "f.bin", size: int = 1) -> Artifact:
    return Artifact(sha256=sha, size=size, name=name)


def test_artifact_shas_walks_nested_containers():
    a, b, c = _file("a" * 64), _file("b" * 64), _file("c" * 64)
    obj = {"x": [a, (b,)], "y": {"deep": c}}  # dict, list, tuple, nested dict
    assert artifact_shas(obj) == {"a" * 64, "b" * 64, "c" * 64}


def test_artifact_shas_reaches_dataclass_fields():
    @dataclass
    class Result:
        art: Artifact
        label: str

    assert artifact_shas(Result(art=_file("d" * 64), label="hi")) == {"d" * 64}


def test_artifact_shas_collects_tree_children_not_manifest():
    """A tree's own sha names its manifest, not a stored blob — only its file children count."""
    kids = (_file("1" * 64, "a.bin"), _file("2" * 64, "sub/b.bin"))
    tree = Artifact(sha256="t" * 64, size=2, name="acts", kind="tree", children=kids)
    shas = artifact_shas(tree)
    assert shas == {"1" * 64, "2" * 64}
    assert "t" * 64 not in shas


def test_artifact_shas_empty_for_plain_values():
    assert artifact_shas({"metrics": [1, 2, 3], "name": "run", "ok": True}) == set()


def test_artifact_shas_prunes_at_callables():
    """The walk stops at code/module boundaries: an Artifact reachable only through a
    function's closure is invisible (crossing it would drag in unrelated module state)."""
    hidden = _file("e" * 64)

    def fn():
        return hidden  # captured as a closure cell — behind a FunctionType

    assert artifact_shas({"fn": fn}) == set()


# ---------------------------------------------------------------------------
# The sidecar: a real step's result artifacts, indexed without unpickling
# ---------------------------------------------------------------------------


def test_sidecar_indexes_result_blobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    app = LocalApparatus("sidecar")
    _drive(_sweep("sidecar", _put_step(), [1]), app)

    store = app.memo_store()
    [rec] = store.records()
    key, gen = rec["key"], rec["gen"]
    art = store.result(key)
    # The sidecar exists next to the result and lists exactly the put blob's sha —
    # so result_artifacts answers without touching the pickle.
    assert store.artifacts_path(key, gen).exists()
    assert store.result_artifacts(key) == [art.sha256]


def test_stale_sidecar_swept_as_attempt_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A sidecar under a replaced generation is unreachable, so the memo sweep collects it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    app = LocalApparatus("stale-side")
    _drive(_sweep("stale-side", _put_step(), [1]), app)
    store = app.memo_store()
    [rec] = store.records()
    (store.result_dir(rec["key"]) / "result-deadbeef.artifacts.json").write_text('["oldsha"]')

    [item] = plan_gc(store).by_kind("attempt-files")
    assert "result-deadbeef.artifacts.json" in item.names


# ---------------------------------------------------------------------------
# collect_store_roots: the mark phase, fail-closed
# ---------------------------------------------------------------------------


def _done_record(memo: MemoStore, key: str, gen: str) -> None:
    memo.records_backend.write(key, {"key": key, "state": "done", "gen": gen})


def test_roots_read_from_sidecar_without_unpickling(tmp_path: Path):
    memo = MemoStore(tmp_path / "exp")
    _done_record(memo, "t-1", "g")
    d = memo.result_dir("t-1")
    d.mkdir(parents=True)
    (d / "result-g.artifacts.json").write_text('["aa", "bb"]')
    # A poisoned pickle proves the sidecar path never reads the result itself.
    (d / "result-g.pkl").write_bytes(b"not a pickle")

    roots, notes = collect_store_roots(stores=[("exp", memo)])
    assert roots == {"aa", "bb"} and notes == []


def test_roots_unpickle_legacy_record_without_sidecar(tmp_path: Path):
    memo = MemoStore(tmp_path / "exp")
    _done_record(memo, "t-1", "g")
    d = memo.result_dir("t-1")
    d.mkdir(parents=True)
    (d / "result-g.pkl").write_bytes(cloudpickle.dumps({"art": _file("f" * 64)}))  # no sidecar

    roots, _ = collect_store_roots(stores=[("exp", memo)])
    assert "f" * 64 in roots


def test_roots_fail_closed_on_in_flight_task(tmp_path: Path):
    memo = MemoStore(tmp_path / "exp")
    memo.records_backend.write("t-1", {"key": "t-1", "state": "running", "gen": "g"})
    with pytest.raises(StoreGcError, match="in flight"):
        collect_store_roots(stores=[("exp", memo)])


def test_roots_fail_closed_on_unreadable_result(tmp_path: Path):
    memo = MemoStore(tmp_path / "exp")
    _done_record(memo, "t-1", "g")
    d = memo.result_dir("t-1")
    d.mkdir(parents=True)
    (d / "result-g.pkl").write_bytes(b"not a pickle")  # DONE, no sidecar, corrupt → references unknown
    with pytest.raises(StoreGcError, match="cannot read the result"):
        collect_store_roots(stores=[("exp", memo)])


def test_roots_fail_closed_on_unknown_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    exp = data_root() / "weird"
    (exp / ".control" / "memo").mkdir(parents=True)
    (exp / ".app").write_text("quantum\n")  # neither local nor modal
    with pytest.raises(StoreGcError, match="unknown backend"):
        collect_store_roots()


def test_roots_note_expired_modal_plane():
    """A Modal control plane whose Dict expired has no records to mark — a note, not an error."""
    roots, notes = collect_store_roots(stores=[("gone", None)])
    assert roots == set()
    assert any("expired" in n for n in notes)


# ---------------------------------------------------------------------------
# plan_store_gc / apply_store_gc: the sweep over a LocalStore
# ---------------------------------------------------------------------------


def test_sweep_keeps_referenced_and_ref_pinned_collects_orphan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    app = LocalApparatus("cas")
    _drive(_sweep("cas", _put_step(), [1]), app)
    store = store_for(data_root() / "store")

    step_art = app.memo_store().result(app.memo_store().records()[0]["key"])
    orphan = store.put(b"orphan bytes", name="orphan.bin")  # no record, no ref
    pinned = store.put(b"pinned bytes", name="pinned.bin")
    store.set_ref("keep/this", pinned)  # the documented pin

    roots, _ = collect_store_roots()
    plan = plan_store_gc(store, roots, grace=0.0, now=time.time() + 1)
    assert _shas(plan) == {orphan.sha256}
    assert plan.referenced >= 2  # the step's result blob and the pinned blob

    apply_store_gc(store, plan)
    assert not store.has(orphan.sha256)
    assert store.has(step_art.sha256) and store.has(pinned.sha256)


def test_grace_window_keeps_young_blobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    store = store_for(data_root() / "store")
    fresh = store.put(b"just written", name="fresh.bin")

    # A wide grace keeps the unreferenced-but-young blob; note explains why.
    plan = plan_store_gc(store, roots=set(), grace=3600.0)
    assert _shas(plan) == set()
    assert plan.in_grace == 1
    assert any("grace window" in n for n in plan.notes)
    # With no grace it becomes collectible.
    plan = plan_store_gc(store, roots=set(), grace=0.0, now=time.time() + 1)
    assert _shas(plan) == {fresh.sha256}


def test_superseded_record_pins_its_blob_until_memo_gc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A superseded record is still a mark root; its blob is collectible only once
    ``mini gc <name>`` removes the record."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    app = LocalApparatus("pin")
    _drive(_sweep("pin", _put_step(), [1, 2]), app)
    _drive(_sweep("pin", _put_step(), [1]), app)  # input 2 removed → its record superseded
    store = store_for(data_root() / "store")
    memo = app.memo_store()

    _, superseded = memo.split_current(memo.records())
    [dead] = superseded
    dead_sha = memo.result(dead["key"]).sha256

    roots, _ = collect_store_roots()
    assert dead_sha in roots  # every record is a root, superseded included
    assert dead_sha not in _shas(plan_store_gc(store, roots, grace=0.0, now=time.time() + 1))

    apply_gc(memo, plan_gc(memo))  # collect the superseded record first
    roots, _ = collect_store_roots()
    assert dead_sha not in roots
    assert dead_sha in _shas(plan_store_gc(store, roots, grace=0.0, now=time.time() + 1))


# ---------------------------------------------------------------------------
# HFStore gc surface (fake api — no network)
# ---------------------------------------------------------------------------


def _hf_entry(path: str, size: int = 10, ts: datetime | None = None):
    return SimpleNamespace(type="file", path=path, size=size, uploaded_at=ts, mtime=None)


class _FakeHFApi:
    """Just the ``HfApi`` surface the gc paths touch: tree listing + delete batches."""

    def __init__(self, tree: dict[str, list] | None = None):
        self.tree = tree or {}
        self.delete_batches: list[list[str]] = []

    def list_bucket_tree(self, bucket, prefix, recursive=True):
        yield from self.tree.get(prefix, [])

    def batch_bucket_files(self, bucket, add=None, delete=None, copy=None):
        if delete is not None:
            self.delete_batches.append(list(delete))


def _hf(tmp_path: Path, api: _FakeHFApi):
    from mini.hf_store import HFStore

    store = HFStore("ns/bkt", cache=LocalStore(tmp_path / "cache"), token="tok")
    store._api = api  # inject the fake; the real api property never runs
    return store


def test_hf_list_blobs_and_refs(tmp_path: Path):
    sha = "a" * 64
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    api = _FakeHFApi(
        {
            "cas": [
                _hf_entry(_cas_key(sha), size=42, ts=ts),
                _hf_entry("cas/ab/short"),  # not a 64-hex sha — ignored
            ],
            "refs": [SimpleNamespace(type="file", path="refs/datasets/tiny.json")],
        }
    )
    store = _hf(tmp_path, api)

    [blob] = list(store.list_blobs())
    assert blob.sha256 == sha and blob.size == 42 and blob.modified_at == ts.timestamp()
    assert store.list_refs() == ["datasets/tiny"]


def test_hf_delete_blobs_batches_and_purges_cache(tmp_path: Path):
    api = _FakeHFApi()
    store = _hf(tmp_path, api)
    shas = [f"{i:064x}" for i in range(501)]
    # Warm-cache one blob so we can prove the purge.
    hot = store._cache._blob_path(shas[0])
    hot.parent.mkdir(parents=True, exist_ok=True)
    hot.write_bytes(b"cached bytes")

    store.delete_blobs(shas)
    assert [len(b) for b in api.delete_batches] == [500, 1]  # one commit per ≤500 chunk
    assert api.delete_batches[0][0] == _cas_key(shas[0])  # sharded key, not the bare sha
    assert not hot.exists()  # cache purged so has() can't lie afterwards


# ---------------------------------------------------------------------------
# Modal per-experiment gc leg (fake Volume + Dict-backed records)
# ---------------------------------------------------------------------------


class _Entry:
    def __init__(self, path: str, is_file: bool = True, size: int = 0):
        self.path = path
        self.type = SimpleNamespace(name="FILE" if is_file else "DIR")
        self.size = size


class _FakeVolume:
    """A ``modal.Volume`` duck: one recursive ``listdir``, per-path ``remove_file``."""

    def __init__(self, entries: list[_Entry]):
        self._entries = entries
        self.removed: list[tuple[str, bool]] = []

    def listdir(self, path: str, recursive: bool = False):
        return list(self._entries)

    def remove_file(self, path: str, recursive: bool = False):
        self.removed.append((path, recursive))


def test_modal_gc_io_reads_memo_tree(tmp_path: Path):
    vol = _FakeVolume(
        [
            _Entry("_memo/task-aaaa", is_file=False),  # dir entry, not a file
            _Entry("_memo/task-aaaa/result-g1.pkl", size=10),
            _Entry("_memo/task-aaaa/result-g1.artifacts.json", size=4),
        ]
    )
    tree = ModalGcIO(vol).memo_tree()
    assert tree["task-aaaa"] == {"result-g1.pkl": 10, "result-g1.artifacts.json": 4}
    assert ModalGcIO(vol).staged_calls() == {}  # Modal passes the call to spawn — nothing staged


def test_modal_gc_plan_and_apply_over_fakes(tmp_path: Path):
    from mini.modal_apparatus import ModalRecordStore

    d: dict = {}
    memo = MemoStore(tmp_path / "vol", records=ModalRecordStore(d))
    memo.records_backend.write("task-aaaa", {"key": "task-aaaa", "state": "done", "gen": "g1"})
    memo.records_backend.write("old-bbbb", {"key": "old-bbbb", "state": "done", "gen": "g2"})
    memo.set_meta(requested=["task-aaaa"], complete=True)  # old-bbbb no longer requested → superseded

    vol = _FakeVolume(
        [
            _Entry("_memo/task-aaaa/result-g1.pkl", size=10),
            _Entry("_memo/old-bbbb/result-g2.pkl", size=8),
            _Entry("_memo/ghost-cccc/result-x.pkl", size=6),  # expired Dict record left an orphan dir
        ]
    )
    io = ModalGcIO(vol)

    plan = plan_gc(memo, memo.records(), io)
    assert {i.key for i in plan.by_kind("superseded")} == {"old-bbbb"}
    assert {i.key for i in plan.by_kind("orphan-dir")} == {"ghost-cccc"}

    apply_gc(memo, plan, io)
    assert ("_memo/old-bbbb", True) in vol.removed  # per-path recursive rm
    assert ("_memo/ghost-cccc", True) in vol.removed
    assert "old-bbbb" not in d  # the Dict record went too
    assert "task-aaaa" in d  # the current record is untouched


# ---------------------------------------------------------------------------
# CLI: mini gc --store
# ---------------------------------------------------------------------------


def _store_ns(**kw) -> argparse.Namespace:
    base = dict(name=None, store=True, apply=False, grace="0d", app=None)
    return argparse.Namespace(**{**base, **kw})


def test_cmd_gc_store_dry_run_then_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    from mini.__main__ import cmd_gc

    app = LocalApparatus("clistore")
    _drive(_sweep("clistore", _put_step(), [1]), app)  # a referenced blob (kept)
    store = store_for(data_root() / "store")
    orphan = store.put(b"orphan for cli", name="o.bin")

    cmd_gc(_store_ns())
    out = capsys.readouterr().out
    assert "dry run" in out and "unreferenced: 1" in out
    assert store.has(orphan.sha256)  # dry run deleted nothing

    cmd_gc(_store_ns(apply=True))
    assert "reclaimed" in capsys.readouterr().out
    assert not store.has(orphan.sha256)

    cmd_gc(_store_ns())  # idempotent: the referenced blob is all that's left
    assert "nothing to collect" in capsys.readouterr().out


def test_cmd_gc_rejects_name_and_store_together():
    from mini.__main__ import cmd_gc

    with pytest.raises(SystemExit, match="not both"):
        cmd_gc(argparse.Namespace(name="x", store=True, apply=False, grace="14d", app=None))

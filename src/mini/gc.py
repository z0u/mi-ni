"""Reclaim storage that no current read path can reach (``mini gc``).

Two sweeps, two scopes:

- **Per-experiment** (``mini gc <name>``): memo-store state — records, result
  dirs, staged calls — on either backend. Locally that's files under
  ``.mini/<name>``; on Modal the records live in a named ``Dict`` (which
  self-expires after 7 idle days) and the result dirs on the Volume (which
  never expire) — the same plan logic runs over a :class:`GcIO` adapter.
- **Project store** (``mini gc --store``): the content-addressed artifact CAS,
  mark-and-sweep. *Mark* walks every experiment's records (both backends) and
  the store's refs; *sweep* deletes blobs nothing reaches. See
  :func:`collect_store_roots` for the safety posture — it fails closed.

Collectibility is judged against the store's own invariants, not age or size:

- A **superseded record** (its key absent from the requested-keys manifest) is
  collectible once the manifest is trustworthy: the last tick ran the DAG to
  completion (``complete`` in the run meta) and nothing is still unsettled.
  A *current* record is never collectible — a DONE one is a future memo hit,
  and even a FAILED one is live state (deleting it would silently convert a
  terminal failure into a relaunch on the next wake).
- A **stale attempt file** (a ``result-<gen>.pkl``/``error-<gen>.txt``/
  ``result-<gen>.artifacts.json`` under a generation the record no longer
  owns) is unreachable: readers resolve through the record's current ``gen``,
  and a fenced zombie writer can't make anything read it again. The one
  exception is the legacy ``error.txt``, which ``MemoStore.error`` still falls
  back to when the current attempt left no traceback — that stays live until
  the current generation writes its own.
- An **orphaned result dir** has no record at all. Records are claimed before
  the worker creates its dir, so this is debris, not a race — on Modal it is
  also the normal end state of a Dict record that expired out from under the
  Volume.
- A **staged call** (``.control/memo/<key>.pkl``) is worker spawn input; it is
  dead once its task is off RUNNING (a relaunch rewrites it). Modal passes the
  call to ``spawn`` directly, so that backend stages nothing.
- An **unreferenced blob** in the CAS is one no record's result and no ref
  reaches, *and* older than the grace window. The window is what makes the
  sweep safe against writers the mark phase cannot see — a checkout that
  hasn't pushed its memo state, or a ``put`` that skipped an upload because
  the blob already existed moments before the sweep judged it garbage.
"""

from __future__ import annotations

import re
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from mini.memo import MemoStore
from mini.runs import SETTLED, RunState, data_root
from mini.store import BlobStat, Store, artifact_shas

__all__ = [
    "GcIO",
    "LocalGcIO",
    "ModalGcIO",
    "GcItem",
    "GcPlan",
    "plan_gc",
    "apply_gc",
    "StoreGcError",
    "StoreGcPlan",
    "collect_store_roots",
    "plan_store_gc",
    "apply_store_gc",
]

# The worker-written files an attempt owns; anything else in a result dir is
# unknown, and unknown is not garbage.
_ATTEMPT_FILE = re.compile(r"result(-\w+)?\.pkl|result(-\w+)?\.artifacts\.json|error(-\w+)?\.txt")

# Keep unreferenced blobs younger than this by default (overridable via
# ``--grace``). Two weeks is git's prune horizon, and comfortably longer than
# any window in which an unseen writer could have referenced an old blob.
GRACE_DEFAULT = "14d"


# ---------------------------------------------------------------------------
# I/O-plane access, backend-shaped
# ---------------------------------------------------------------------------


class GcIO(ABC):
    """The I/O-plane operations a memo sweep needs, decoupled from where files live.

    One listing up front (``memo_tree``), deletes addressed by key + file name —
    so the same plan/apply logic serves local disk and a Modal Volume (per-path
    ``remove_file``; no local mount).
    """

    @abstractmethod
    def memo_tree(self) -> dict[str, dict[str, int]]:
        """``{key: {filename: size}}`` for every result dir under ``_memo/``.

        Nested files appear under their relative path (``sub/x.bin``) — they
        count toward sizes but never match the attempt-file pattern, so unknown
        content is sized, not swept.
        """

    @abstractmethod
    def staged_calls(self) -> dict[str, int]:
        """``{key: size}`` of staged spawn inputs; empty where calls aren't staged."""

    @abstractmethod
    def delete_dir(self, key: str) -> None:
        """Remove a whole result dir (idempotent)."""

    @abstractmethod
    def delete_files(self, key: str, names: list[str]) -> None:
        """Remove named files within a result dir (idempotent)."""

    @abstractmethod
    def delete_call(self, key: str) -> None:
        """Remove a staged call (idempotent)."""


class LocalGcIO(GcIO):
    """Plain-filesystem I/O plane: the memo store's own ``data_dir``."""

    def __init__(self, store: MemoStore):
        self._store = store

    def memo_tree(self) -> dict[str, dict[str, int]]:
        root = self._store.data_dir / "_memo"
        if not root.is_dir():
            return {}
        return {
            d.name: {p.relative_to(d).as_posix(): p.stat().st_size for p in sorted(d.rglob("*")) if p.is_file()}
            for d in sorted(root.iterdir())
            if d.is_dir()
        }

    def staged_calls(self) -> dict[str, int]:
        root = self._store.root
        if not root.is_dir():
            return {}
        return {p.stem: p.stat().st_size for p in sorted(root.glob("*.pkl"))}

    def delete_dir(self, key: str) -> None:
        shutil.rmtree(self._store.result_dir(key), ignore_errors=True)

    def delete_files(self, key: str, names: list[str]) -> None:
        for name in names:
            (self._store.result_dir(key) / name).unlink(missing_ok=True)

    def delete_call(self, key: str) -> None:
        self._store._call(key).unlink(missing_ok=True)


class ModalGcIO(GcIO):
    """Modal-Volume I/O plane: one recursive ``listdir``, per-path ``remove_file``.

    *volume* is a ``modal.Volume`` (or any duck with ``listdir``/``remove_file``
    — a fake for tests). Import-light on purpose: constructing this never touches
    the network; the first listing does.
    """

    def __init__(self, volume: Any):
        self._vol = volume

    def memo_tree(self) -> dict[str, dict[str, int]]:
        import modal

        tree: dict[str, dict[str, int]] = {}
        try:
            entries = list(self._vol.listdir("_memo", recursive=True))
        except FileNotFoundError, modal.exception.NotFoundError:
            return {}
        for e in entries:
            parts = Path(e.path).parts  # '_memo/<key>/<file...>'
            if parts[:1] != ("_memo",) or len(parts) < 2:
                continue
            key = parts[1]
            entry = tree.setdefault(key, {})
            if len(parts) > 2 and getattr(e.type, "name", "") == "FILE":
                entry["/".join(parts[2:])] = getattr(e, "size", 0) or 0
        return tree

    def staged_calls(self) -> dict[str, int]:
        return {}  # Modal passes the call straight to spawn — nothing staged

    def delete_dir(self, key: str) -> None:
        self._remove(f"_memo/{key}", recursive=True)

    def delete_files(self, key: str, names: list[str]) -> None:
        for name in names:
            self._remove(f"_memo/{key}/{name}")

    def delete_call(self, key: str) -> None:
        pass

    def _remove(self, path: str, recursive: bool = False) -> None:
        import modal

        try:
            self._vol.remove_file(path, recursive=recursive)
        except FileNotFoundError, modal.exception.NotFoundError:
            pass  # already gone — deletes are idempotent


# ---------------------------------------------------------------------------
# Per-experiment sweep
# ---------------------------------------------------------------------------


@dataclass
class GcItem:
    kind: str  # 'superseded' | 'attempt-files' | 'orphan-dir' | 'staged-call'
    key: str
    names: list[str]  # files involved, for display; deletes go through GcIO by kind
    size: int


@dataclass
class GcPlan:
    items: list[GcItem] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)  # reasons collectible-looking state was left alone

    @property
    def size(self) -> int:
        return sum(i.size for i in self.items)

    def by_kind(self, kind: str) -> list[GcItem]:
        return [i for i in self.items if i.kind == kind]


def plan_gc(store: MemoStore, records: list[dict] | None = None, io: GcIO | None = None) -> GcPlan:
    """What ``apply_gc`` would delete, and why the rest stays.

    Call ``reap_dead`` first so a vanished worker's RUNNING record doesn't read
    as alive. Pass *records* to reuse a snapshot already in hand; pass *io* to
    plan against a non-local I/O plane (a Modal Volume).
    """
    records = store.records() if records is None else records
    io = io or LocalGcIO(store)
    tree = io.memo_tree()
    calls = io.staged_calls()
    plan = GcPlan()
    collected = _plan_superseded(store, records, tree, calls, plan)
    _plan_attempt_files(store, records, tree, collected, plan)
    _plan_orphan_dirs(records, tree, plan)
    _plan_staged_calls(records, calls, collected, plan)
    return plan


def _plan_superseded(
    store: MemoStore, records: list[dict], tree: dict[str, dict[str, int]], calls: dict[str, int], plan: GcPlan
) -> set[str]:
    """Whole superseded records — record, result dir, and staged call — gate permitting."""
    current, superseded = store.split_current(records)
    if not superseded:
        return set()
    unsettled = [r for r in current if r.get("state") not in SETTLED]
    if not store.meta().get("complete"):
        plan.kept.append(
            f"{len(superseded)} superseded record(s): the last tick did not run the DAG to "
            "completion, so the manifest may be missing keys a later stage still wants"
        )
        return set()
    if unsettled:
        plan.kept.append(f"{len(superseded)} superseded record(s): {len(unsettled)} task(s) still unsettled")
        return set()
    collected: set[str] = set()
    for rec in superseded:
        key = rec["key"]
        if rec.get("state") == RunState.RUNNING:  # reap_dead left it: the worker is provably alive
            plan.kept.append(f"{key}: superseded, but its worker is still alive — cancel it first")
            continue
        size = sum(tree.get(key, {}).values()) + calls.get(key, 0)
        plan.items.append(GcItem("superseded", key, sorted(tree.get(key, {})), size))
        collected.add(key)
    return collected


def _plan_attempt_files(
    store: MemoStore, records: list[dict], tree: dict[str, dict[str, int]], collected: set[str], plan: GcPlan
) -> None:
    """Attempt files no read through the record's current gen can reach."""
    for rec in records:
        key = rec["key"]
        files = tree.get(key)
        if key in collected or not files:
            continue
        gen = rec.get("gen")
        live = {
            store.result_path(key, gen).name,
            store.error_path(key, gen).name,
            store.artifacts_path(key, gen).name,
        }
        if gen and store.error_path(key, gen).name not in files:
            live.add(store.error_path(key, None).name)  # error() falls back to the legacy name
        stale = [n for n in sorted(files) if _ATTEMPT_FILE.fullmatch(n) and n not in live]
        if stale:
            plan.items.append(GcItem("attempt-files", key, stale, sum(files[n] for n in stale)))


def _plan_orphan_dirs(records: list[dict], tree: dict[str, dict[str, int]], plan: GcPlan) -> None:
    """Result dirs with no record at all (e.g. a control plane that expired out from under the volume)."""
    known = {r["key"] for r in records}
    for key in sorted(tree):
        if key not in known:
            plan.items.append(GcItem("orphan-dir", key, sorted(tree[key]), sum(tree[key].values())))


def _plan_staged_calls(records: list[dict], calls: dict[str, int], collected: set[str], plan: GcPlan) -> None:
    """Cloudpickled spawn inputs for tasks that are no longer running."""
    recs = {r["key"]: r for r in records}
    for key, size in sorted(calls.items()):
        if key in collected:
            continue
        if (recs.get(key) or {}).get("state") != RunState.RUNNING:
            plan.items.append(GcItem("staged-call", key, [f"{key}.pkl"], size))


def apply_gc(store: MemoStore, plan: GcPlan, io: GcIO | None = None) -> None:
    """Delete everything in *plan*.

    Record first, files second: a crash between the two leaves an orphaned dir,
    which the next gc collects — never the reverse (a record whose result dir
    is gone).
    """
    io = io or LocalGcIO(store)
    for item in plan.items:
        if item.kind == "superseded":
            store.records_backend.delete(item.key)
            io.delete_dir(item.key)
            io.delete_call(item.key)
        elif item.kind == "attempt-files":
            io.delete_files(item.key, item.names)
        elif item.kind == "orphan-dir":
            io.delete_dir(item.key)
        elif item.kind == "staged-call":
            io.delete_call(item.key)


# ---------------------------------------------------------------------------
# Project-store (CAS) mark-and-sweep
# ---------------------------------------------------------------------------


class StoreGcError(RuntimeError):
    """The sweep cannot establish a safe reference set — fail closed, delete nothing."""


@dataclass
class StoreGcPlan:
    unreferenced: list[BlobStat] = field(default_factory=list)  # what apply_store_gc deletes
    total_blobs: int = 0
    total_size: int = 0
    referenced: int = 0
    in_grace: int = 0
    roots: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(b.size for b in self.unreferenced)


def _experiment_names(root: Path) -> list[str]:
    """Experiments with any durable trace under the data root.

    ``.control/memo`` marks a local memo store; a bare ``.app`` stamp marks an
    experiment whose state lives on another backend (Modal). ``store/``,
    ``store-cache/`` and ``exports/`` carry neither, so they never read as
    experiments.
    """
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / ".control" / "memo").is_dir() or (p / ".app").is_file())


def _memo_store_for(name: str, root: Path) -> MemoStore | None:
    """The memo store for *name* on its stamped backend — ``None`` if the Modal
    control plane no longer exists (expired Dict: no records, so no roots).
    """
    marker = root / name / ".app"
    backend = marker.read_text().strip() if marker.is_file() else "local"
    if backend in ("local", ""):
        return MemoStore(root / name)
    if backend != "modal":
        raise StoreGcError(f"{name}: unknown backend {backend!r} stamped in {marker} — cannot mark its references")
    import modal

    from mini.modal_apparatus import ModalMemoStore, ModalRecordStore, control_dict_name
    from mini.modal_volume import ModalVolume

    d = modal.Dict.from_name(control_dict_name(name))  # no create_if_missing: marking must not mint state
    try:
        d.hydrate()
    except modal.exception.NotFoundError:
        return None
    return ModalMemoStore(ModalVolume(name, create=False), ModalRecordStore(d))


def collect_store_roots(
    root: Path | None = None, stores: Iterable[tuple[str, MemoStore | None]] | None = None
) -> tuple[set[str], list[str]]:
    """Mark phase: every blob sha any experiment's records can still reach.

    Fails closed (:class:`StoreGcError`) rather than under-marking:

    - **An in-flight task blocks the sweep entirely.** A running worker may
      have just seen ``has(sha) == True`` for bytes it is about to reference —
      deleting that blob would corrupt the result it hasn't written yet.
    - **An unreadable result blocks the sweep.** A record without an artifact
      sidecar must be unpickled to learn its references; if that fails (moved
      code, missing volume), its references are unknown — so nothing is safe.

    *Every* record present is a root — superseded ones included. Collecting a
    superseded record's blobs is ``mini gc <name>``'s call to make first; the
    store sweep never second-guesses the memo layer.

    Pass *stores* to mark an explicit ``(name, memo_store)`` set (tests);
    otherwise experiments are enumerated under *root* and each is read on the
    backend stamped at launch.
    """
    root = data_root() if root is None else root
    if stores is None:
        stores = ((name, _memo_store_for(name, root)) for name in _experiment_names(root))
    shas: set[str] = set()
    notes: list[str] = []
    for name, memo in stores:
        if memo is None:
            notes.append(f"{name}: Modal control plane expired or absent — no records to mark")
            continue
        for rec in memo.records():
            key = rec["key"]
            if rec.get("state") in (RunState.RUNNING, RunState.PENDING):
                raise StoreGcError(
                    f"{name}/{key} is in flight — its worker may be about to reference blobs this sweep "
                    f"would judge unreachable. Let it settle (or reap it: mini status {name}), then re-run."
                )
            listed = memo.result_artifacts(key)
            if listed is not None:
                shas.update(listed)
            elif rec.get("state") == RunState.DONE:  # pre-sidecar record: the result itself is the index
                try:
                    shas.update(artifact_shas(memo.result(key)))
                except Exception as exc:
                    raise StoreGcError(
                        f"cannot read the result of {name}/{key} to learn which blobs it references "
                        f"({type(exc).__name__}: {exc}) — repair it or collect the record first (mini gc {name})"
                    ) from exc
    return shas, notes


def plan_store_gc(store: Store, roots: set[str], *, grace: float, now: float | None = None) -> StoreGcPlan:
    """Sweep phase: every blob outside *roots* ∪ refs and older than *grace* seconds.

    Refs are resolved here (they live in the store itself), so a blob pinned by
    ``set_ref`` survives even with no record referencing it — that's the
    documented way to keep an artifact alive across record gc. A blob younger
    than *grace* (or of unknown age) is never collected: it may belong to a
    writer the mark phase couldn't see.
    """
    now = time.time() if now is None else now
    roots = set(roots)
    for name in store.list_refs():
        if (art := store.get_ref(name)) is not None:
            roots.update(artifact_shas(art))
    plan = StoreGcPlan(roots=len(roots))
    for blob in store.list_blobs():
        plan.total_blobs += 1
        plan.total_size += blob.size
        if blob.sha256 in roots:
            plan.referenced += 1
        elif blob.modified_at is None or now - blob.modified_at < grace:
            plan.in_grace += 1
        else:
            plan.unreferenced.append(blob)
    if plan.in_grace:
        plan.notes.append(f"{plan.in_grace} unreferenced blob(s) kept: younger than the grace window")
    return plan


def apply_store_gc(store: Store, plan: StoreGcPlan) -> None:
    """Delete every blob in *plan* (and purge any warm cache of them)."""
    store.delete_blobs([b.sha256 for b in plan.unreferenced])

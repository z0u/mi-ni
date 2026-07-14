"""
Detached worker for a single memoized task: ``python -m mini._taskworker <data_dir> <key>``.

Loads the cloudpickled call, runs it with the data-dir + progress context (so
``get_data_dir``/``emit_progress``/``emit_metrics`` work), and records the
result or traceback under the content key. Spawned in its own session so it
outlives the orchestration tick that launched it.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import cloudpickle

from contextlib import nullcontext

from mini._queues import EndOfQueue
from mini.memo import MemoStore
from mini.progress import progress_context
from mini.runs import RunState, compute_env
from mini.store import (
    Artifact,
    StaleWriteError,
    Store,
    artifact_shas,
    producer_context,
    resolved_refs_context,
    store_context,
    store_for,
    store_root_for,
)
from mini.volume import data_dir_context


class _MemoSink:
    """Writes the latest progress/metrics for a task straight to its memo record.

    Writes are fenced on the attempt's *gen*: once a successor attempt (or a
    ``cancel``) takes the record, this worker's progress stops landing — and stops
    trying (``_fenced``), since a superseded attempt can never own it again.
    """

    def __init__(self, store: MemoStore, key: str, gen: str | None = None):
        self._store = store
        self._key = key
        self._gen = gen
        self._fenced = False

    def put(self, item: Any, /, block: bool = True, timeout: float | None = None) -> None:
        del block, timeout
        if isinstance(item, EndOfQueue) or self._fenced:
            return
        fields = dict(
            step=item.step,
            total=item.total,
            message=item.message,
            metrics=item.metrics,
            heartbeat_at=time.time(),
        )
        if self._gen is None:
            self._store.update(self._key, **fields)
        elif not self._store.update_if(self._key, self._gen, **fields):
            self._fenced = True

    def get(self, /, block: bool = True, timeout: float | None = None) -> Any:
        del block, timeout
        raise NotImplementedError

    def empty(self) -> bool:
        return True


class _FencedStore(Store):
    """The ambient store for one attempt, with mutable-name writes gen-fenced.

    Record writes and results are already fenced on the attempt generation, but
    ``set_ref`` / ``publish`` mutate *names* in the artifact store — unfenced,
    a stale worker's name write would silently last-writer-win its successor's
    (CAS blobs are immune, so everything else passes straight through). The name
    lives in a different backend than the record (files/HF vs the record store),
    so the fence is check → write → re-check rather than atomic: a supersession
    landing *during* the write can't be prevented, but the re-check turns it from
    silent corruption into a loud :class:`~mini.store.StaleWriteError` — and the
    successor's own completing write then heals the name.
    """

    def __init__(self, inner: Store, memo: MemoStore, key: str, gen: str):
        self._inner, self._memo, self._key, self._gen = inner, memo, key, gen

    def _fence(self, verb: str, after: bool = False) -> None:
        if self._memo.record(self._key).get("gen") != self._gen:
            raise StaleWriteError(
                f"{verb}: attempt {self._gen} of task {self._key} was superseded (relaunched or cancelled)"
                + (" during the write — the name may briefly hold this attempt's value" if after else "")
            )

    # Fenced mutable-name verbs — everything else passes through untouched.

    def set_ref(self, name: str, art: Artifact) -> None:
        self._fence(f"set_ref({name!r})")
        self._inner.set_ref(name, art)
        self._fence(f"set_ref({name!r})", after=True)

    def publish(self, art: Artifact, path: str) -> str:
        self._fence(f"publish({path!r})")
        url = self._inner.publish(art, path)
        self._fence(f"publish({path!r})", after=True)
        return url

    def _write_ref(self, name: str, payload: str) -> None:
        self._fence(f"set_ref({name!r})")
        self._inner._write_ref(name, payload)
        self._fence(f"set_ref({name!r})", after=True)

    # Pass-throughs: forward the public verbs (not the shared high-level logic)
    # so a backend's own overrides — HF batching, cache warming — stay in play.

    def put(self, data: bytes | Path, *, name: str) -> Artifact:
        return self._inner.put(data, name=name)

    def get(self, art: Artifact, dest: Path) -> Path:
        return self._inner.get(art, dest)

    def get_ref(self, name: str) -> Artifact | None:
        return self._inner.get_ref(name)

    def has(self, sha256: str) -> bool:
        return self._inner.has(sha256)

    def _write_blob(self, sha256: str, src: Path) -> None:
        self._inner._write_blob(sha256, src)

    def _read_blob(self, sha256: str, dest: Path) -> None:
        self._inner._read_blob(sha256, dest)

    def _read_ref(self, name: str) -> str | None:
        return self._inner._read_ref(name)


def _producer_stamp(experiment: str | None, store: MemoStore, key: str) -> dict[str, Any] | None:
    """The identity ``set_ref`` stamps into refs this task writes, or ``None``.

    A compact provenance record (the :func:`~mini.lineage.upstream_snapshot` shape
    plus the task key): enough for a consumer — a downstream run, a report — to
    attribute the bytes to this experiment and the code state that produced them,
    without a lookup into this run's control plane. The code state comes from the
    run's stored lineage (stamped by the driver at wake start); best-effort, since
    provenance must never take a task down.
    """
    if experiment is None:
        return None
    from mini.lineage import upstream_snapshot

    try:
        meta = store.meta()
    except Exception:
        meta = {}
    stamp = upstream_snapshot(experiment, meta)
    stamp.pop("modal_app_ids", None)  # app ids accumulate run-wide; they don't identify this write
    stamp["task"] = key
    return stamp


def _upstream_refs(resolved: dict[str, dict[str, Any] | None]) -> list[dict[str, str]]:
    """Compact ``{ref, experiment?}`` entries for the record, from the resolved-ref set."""
    return [
        {"ref": name, **({"experiment": p["experiment"]} if p and p.get("experiment") else {})}
        for name, p in sorted(resolved.items())
    ]


def execute_task(
    store: MemoStore,
    key: str,
    fn: Any,
    args: tuple,
    hooks: list,
    commit: Callable[[], None] | None = None,
    artifacts: Store | None = None,
    gen: str | None = None,
    experiment: str | None = None,
) -> None:
    """Run one memoized call and persist its result/state — backend-agnostic.

    Shared by the local subprocess worker and the Modal remote worker: only how
    the call *arrives* (staged on disk vs passed to ``spawn``) and where state
    lands (``RecordStore``) differ; the run/persist core is identical.

    *gen* is the attempt generation this worker runs under. Every record write is
    fenced on it, and the result/error land in gen-qualified files — so a stale
    worker (superseded by a relaunch, or cancelled but surviving SIGTERM) can
    neither merge DONE over its successor's RUNNING nor overwrite its result. A
    worker that finds itself already superseded at startup exits without running.

    *commit* is called after the result/error is written to the I/O plane and
    *before* the record flips to DONE/FAILED — so a poller never sees a settled
    state whose artifact hasn't been committed yet (the Modal Volume needs this).

    *artifacts* binds the content-addressed :class:`~mini.store.Store` as ambient
    for ``mini.store.put`` / ``get`` inside the step. Because ``put`` uploads
    synchronously, by the time the result is written its handles already resolve —
    so the existing write → commit → DONE order extends from "the volume flushed"
    to "the referenced blobs are durable" for free. Its mutable-name verbs
    (``set_ref`` / ``publish``) are fenced on *gen* via :class:`_FencedStore`, so
    a stale worker fails loudly instead of clobbering a name its successor owns.

    *experiment* is the experiment this task belongs to. It powers ref provenance
    both ways: refs the step writes are stamped with it (:func:`_producer_stamp`),
    and refs the step *resolves* land on the settled record as ``upstream_refs`` —
    the evidence the driver aggregates into ``lineage.upstreams`` without the
    experiment declaring its deps by hand.
    """

    def record(**fields: Any) -> bool:
        if gen is None:
            store.update(key, **fields)
            return True
        return store.update_if(key, gen, **fields)

    result_dir = store.result_dir(key)
    result_dir.mkdir(parents=True, exist_ok=True)
    sink = _MemoSink(store, key, gen)
    if artifacts is not None and gen is not None:
        artifacts = _FencedStore(artifacts, store, key, gen)
    # Record what we actually ran on (host/GPU/…) and when the worker truly began,
    # captured here in the worker. ``started_at`` (worker's first write) pairs with
    # ``finished_at`` (below) for a real execution duration — distinct from the
    # client-side ``created_at`` stamped before the worker was even scheduled.
    started_at = time.time()
    if not record(state=RunState.RUNNING, heartbeat_at=started_at, started_at=started_at, env=compute_env()):
        return  # superseded before we even started — nothing here is wanted anymore
    producer = _producer_stamp(experiment, store, key)
    resolved: dict[str, dict[str, Any] | None] = {}
    try:
        with (
            data_dir_context(store.data_dir),
            store_context(artifacts) if artifacts is not None else nullcontext(),
            producer_context(producer) if producer is not None else nullcontext(),
            resolved_refs_context(resolved),
            progress_context(key, key, queue=sink, emission_interval=0.2),
        ):
            for hook in reversed(hooks):
                hook()
            result = fn(*args)
        # The artifact sidecar rides with the result: which blobs the result
        # references, written even when empty ("none" beats "unknown" — the GC
        # mark phase then never has to unpickle this result). Sidecar first, so
        # a readable result always has its reference set alongside.
        store.artifacts_path(key, gen).write_text(json.dumps(sorted(artifact_shas(result))))
        store.result_path(key, gen).write_bytes(cloudpickle.dumps(result))
        if commit is not None:
            commit()
        now = time.time()
        # Which shared refs this step resolved (and whose experiment wrote each) —
        # the per-task evidence the driver rolls up into ``lineage.upstreams``.
        extra = {"upstream_refs": _upstream_refs(resolved)} if resolved else {}
        record(state=RunState.DONE, heartbeat_at=now, finished_at=now, **extra)
    except Exception as exc:
        tb = traceback.format_exc()
        store.error_path(key, gen).write_text(tb)
        if commit is not None:
            commit()
        now = time.time()
        extra = {"upstream_refs": _upstream_refs(resolved)} if resolved else {}
        record(
            state=RunState.FAILED,
            error=tb.strip().splitlines()[-1],
            exc_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            heartbeat_at=now,
            finished_at=now,
            **extra,
        )


def run_task(data_dir: Path, key: str) -> None:
    """Local subprocess entry: read the staged call from disk and run it."""
    store = MemoStore(data_dir)
    fn, args, hooks, gen = store.read_call(key)
    # Project-scoped artifact store sits beside the experiment's data dir (or the
    # shared HF bucket, if MINI_STORE_BUCKET is set), so a blob put here resolves
    # from any experiment in the project (and from reports).
    artifacts = store_for(store_root_for(data_dir))
    # The local data dir is <data_root>/<experiment>, so its leaf names the experiment.
    execute_task(store, key, fn, args, hooks, artifacts=artifacts, gen=gen, experiment=data_dir.name)


def main() -> None:
    run_task(Path(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()

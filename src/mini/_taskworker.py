"""
Detached worker for a single memoized task: ``python -m mini._taskworker <data_dir> <key>``.

Loads the cloudpickled call, runs it with the data-dir + progress context (so
``get_data_dir``/``emit_progress``/``emit_metrics`` work), and records the
result or traceback under the content key. Spawned in its own session so it
outlives the orchestration tick that launched it.
"""

from __future__ import annotations

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
from mini.store import Store, store_context, store_for, store_root_for
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


def execute_task(
    store: MemoStore,
    key: str,
    fn: Any,
    args: tuple,
    hooks: list,
    commit: Callable[[], None] | None = None,
    artifacts: Store | None = None,
    gen: str | None = None,
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
    to "the referenced blobs are durable" for free.
    """

    def record(**fields: Any) -> bool:
        if gen is None:
            store.update(key, **fields)
            return True
        return store.update_if(key, gen, **fields)

    result_dir = store.result_dir(key)
    result_dir.mkdir(parents=True, exist_ok=True)
    sink = _MemoSink(store, key, gen)
    # Record what we actually ran on (host/GPU/…), captured here in the worker.
    if not record(state=RunState.RUNNING, heartbeat_at=time.time(), env=compute_env()):
        return  # superseded before we even started — nothing here is wanted anymore
    try:
        with (
            data_dir_context(store.data_dir),
            store_context(artifacts) if artifacts is not None else nullcontext(),
            progress_context(key, key, queue=sink, emission_interval=0.2),
        ):
            for hook in reversed(hooks):
                hook()
            result = fn(*args)
        store.result_path(key, gen).write_bytes(cloudpickle.dumps(result))
        if commit is not None:
            commit()
        record(state=RunState.DONE, heartbeat_at=time.time())
    except Exception as exc:
        tb = traceback.format_exc()
        store.error_path(key, gen).write_text(tb)
        if commit is not None:
            commit()
        record(
            state=RunState.FAILED,
            error=tb.strip().splitlines()[-1],
            exc_type=f'{type(exc).__module__}.{type(exc).__qualname__}',
            heartbeat_at=time.time(),
        )


def run_task(data_dir: Path, key: str) -> None:
    """Local subprocess entry: read the staged call from disk and run it."""
    store = MemoStore(data_dir)
    fn, args, hooks, gen = store.read_call(key)
    # Project-scoped artifact store sits beside the experiment's data dir (or the
    # shared HF bucket, if MINI_STORE_BUCKET is set), so a blob put here resolves
    # from any experiment in the project (and from reports).
    artifacts = store_for(store_root_for(data_dir))
    execute_task(store, key, fn, args, hooks, artifacts=artifacts, gen=gen)


def main() -> None:
    run_task(Path(sys.argv[1]), sys.argv[2])


if __name__ == '__main__':
    main()

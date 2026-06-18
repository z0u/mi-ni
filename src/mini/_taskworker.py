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
from typing import Any

import cloudpickle

from mini._queues import EndOfQueue
from mini.memo import MemoStore
from mini.progress import progress_context
from mini.runs import RunState
from mini.volume import data_dir_context


class _MemoSink:
    """Writes the latest progress/metrics for a task straight to its memo record."""

    def __init__(self, store: MemoStore, key: str):
        self._store = store
        self._key = key

    def put(self, item: Any, /, block: bool = True, timeout: float | None = None) -> None:
        del block, timeout
        if isinstance(item, EndOfQueue):
            return
        self._store.update(
            self._key,
            step=item.step,
            total=item.total,
            message=item.message,
            metrics=item.metrics,
            heartbeat_at=time.time(),
        )

    def get(self, /, block: bool = True, timeout: float | None = None) -> Any:
        del block, timeout
        raise NotImplementedError

    def empty(self) -> bool:
        return True


def run_task(data_dir: Path, key: str) -> None:
    store = MemoStore(data_dir)
    fn, args, hooks = store.read_call(key)
    result_dir = store.result_dir(key)
    result_dir.mkdir(parents=True, exist_ok=True)
    sink = _MemoSink(store, key)
    store.update(key, state=RunState.RUNNING, heartbeat_at=time.time())
    try:
        with data_dir_context(data_dir), progress_context(key, key, queue=sink, emission_interval=0.2):
            for hook in reversed(hooks):
                hook()
            result = fn(*args)
        (result_dir / 'result.pkl').write_bytes(cloudpickle.dumps(result))
        store.update(key, state=RunState.DONE, heartbeat_at=time.time())
    except Exception:
        tb = traceback.format_exc()
        (result_dir / 'error.txt').write_text(tb)
        store.update(key, state=RunState.FAILED, error=tb.strip().splitlines()[-1], heartbeat_at=time.time())


def main() -> None:
    run_task(Path(sys.argv[1]), sys.argv[2])


if __name__ == '__main__':
    main()

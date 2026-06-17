"""
Detached worker: runs a submitted run's jobs and records durable state.

Spawned as ``python -m mini._worker <data_dir> <run_id>`` in its own session
(so it outlives the launcher). It loads the cloudpickled spec, runs each PENDING
job in a thread pool, and writes progress/metrics/results/errors to the control
and I/O planes — the same state a fresh ``status``/``results`` call reads back.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cloudpickle

from mini._queues import EndOfQueue
from mini.progress import progress_context
from mini.runs import Run, RunState, _run_at
from mini.volume import data_dir_context


class _ControlPlaneSink:
    """A QueueLike that writes the latest progress straight to the control plane.

    Reusing the progress plumbing means user code calls ``emit_progress`` /
    ``emit_metrics`` unchanged; only the sink is durable. Last-writer-wins, so
    it can never fill — unlike a queue left behind by an abandoned run.
    """

    def __init__(self, run: Run, job_id: str):
        self._run = run
        self._job_id = job_id

    def put(self, item: Any, /, block: bool = True, timeout: float | None = None) -> None:
        del block, timeout
        if isinstance(item, EndOfQueue):
            return
        self._run.cp.write_job(
            self._run.id,
            self._job_id,
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


def _run_job(run: Run, fn, args: tuple, kwargs: dict, hooks: list, job_id: str) -> None:
    job_dir = run._job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    sink = _ControlPlaneSink(run, job_id)
    run.cp.write_job(run.id, job_id, state=RunState.RUNNING, heartbeat_at=time.time())
    try:
        with data_dir_context(run.data_dir), progress_context(run.id, job_id, queue=sink, emission_interval=0.2):
            for hook in reversed(hooks):
                hook()
            result = fn(*args, **kwargs)
        (job_dir / 'result.pkl').write_bytes(cloudpickle.dumps(result))
        run.cp.write_job(run.id, job_id, state=RunState.DONE, heartbeat_at=time.time())
    except Exception:
        tb = traceback.format_exc()
        (job_dir / 'error.txt').write_text(tb)
        run.cp.write_job(
            run.id, job_id, state=RunState.FAILED, error=tb.strip().splitlines()[-1], heartbeat_at=time.time()
        )


def run_detached(data_dir: Path, run_id: str) -> None:
    run = _run_at(data_dir, run_id)
    spec = run.cp.read_spec(run_id)
    fn, jobs, kwargs = spec['fn'], spec['jobs'], spec['kwargs']
    hooks, max_workers = spec.get('before_hooks', []), max(1, spec.get('max_workers', 1))

    run.cp.write_run(run_id, state=RunState.RUNNING, pid=os.getpid())
    pending = [str(j) for j in range(len(jobs)) if run.cp.read_job(run_id, str(j)).state == RunState.PENDING]
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(lambda jid: _run_job(run, fn, tuple(jobs[int(jid)]), kwargs, hooks, jid), pending))
    finally:
        run.cp.write_run(run_id, state=run.state())


def main() -> None:
    run_detached(Path(sys.argv[1]), sys.argv[2])


if __name__ == '__main__':
    main()

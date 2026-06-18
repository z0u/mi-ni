"""
Detached worker: runs a submitted run's jobs and records durable state.

This is a *process entrypoint*, not a per-call wrapper. ``LocalApparatus.submit``
spawns it as ``python -m mini._worker <data_dir> <run_id>`` in its own session,
then returns; the worker outlives the launcher and is what a later
``status``/``results`` call reads back through the control and I/O planes.

``_run_job`` is the durable analogue of ``_wrap_for_local`` /
``_wrap_for_modal``: it reuses the same progress/hooks plumbing, but instead of
yielding a result in-process it persists state/metrics/results/errors so they
survive the launcher's death. The two-plane split (see notes/agentic-experiments.md):

  - control plane (small, hot): per-job state/step/metrics/heartbeat, written
    through ``run.cp`` — what ``status`` polls.
  - I/O plane (large, cold): ``result.pkl`` / ``error.txt`` under the job dir —
    what ``results`` / ``logs`` read.
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

"""
Proof-of-concept: decoupled submit / poll / gather on top of mini's primitives.

The point: an agent (Claude Code on the web) cannot hold a process open for the
length of a training run. So the run lifecycle must survive process death and be
*pollable* from a fresh process. This PoC demonstrates that with local compute
only, by splitting the lifecycle into separate CLI invocations:

    python poc.py launch          -> spawns detached workers, prints RUN_ID, exits
    python poc.py poll   RUN_ID    -> reads durable status, prints, exits
    python poc.py gather RUN_ID    -> reads durable results, prints, exits

Workers are detached subprocesses (start_new_session=True) that outlive the
launcher. Each worker reuses mini's *unchanged* emit_progress() — only the sink
is swapped for one that writes status.json durably. That is the whole thesis:
the existing progress plumbing is already sink-agnostic; durability is a backend.
"""

from __future__ import annotations

import json
import os
import pickle
import secrets
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from mini.progress import ProgressMessage, emit_progress, progress_context
from mini._queues import EndOfQueue, QueueLike

ROOT = Path(os.environ.get('POC_ROOT', '/tmp/agentic_poc/runs'))


# --- The experiment itself: a plain, importable function (no notebook state) ---
def train(cfg: dict) -> dict:
    """A toy 'training' job: emit progress, then return a 'metric'."""
    steps = cfg['steps']
    loss = 10.0
    for i in range(steps):
        time.sleep(cfg.get('step_seconds', 0.2))
        loss *= 0.92  # pretend we're learning
        emit_progress(i + 1, steps, message=f'lr={cfg["lr"]:.0e} loss={loss:.3f}')
        if cfg.get('crash_at') == i:
            raise RuntimeError(f'injected failure at step {i}')
    return {'lr': cfg['lr'], 'final_loss': round(loss, 4)}


# --- Durable run store: status.json / result.pkl / error.txt per job ----------
@dataclass
class JobStatus:
    job_id: str
    state: str  # PENDING | RUNNING | DONE | FAILED
    step: int = 0
    total: int = 0
    message: str = ''
    error: str | None = None


class FileStatusSink(QueueLike[ProgressMessage]):
    """A QueueLike that durably writes the latest progress to status.json.

    Reusing the QueueLike protocol means mini's progress_context/emit_progress
    work against it with zero changes to the experiment code.
    """

    def __init__(self, job_dir: Path):
        self._job_dir = job_dir

    def put(self, item, /, block=True, timeout=None) -> None:
        if isinstance(item, EndOfQueue):
            return
        _merge_status(self._job_dir, step=item.step, total=item.total, message=item.message, state='RUNNING')

    def get(self, /, block=True, timeout=None):  # not used on the worker side
        raise NotImplementedError

    def empty(self) -> bool:
        return True


def _merge_status(job_dir: Path, **fields) -> None:
    """Atomically update status.json (read-modify-write, tmp+rename)."""
    path = job_dir / 'status.json'
    cur = json.loads(path.read_text()) if path.exists() else {'job_id': job_dir.name}
    cur.update({k: v for k, v in fields.items() if v is not None or k == 'error'})
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(cur))
    tmp.replace(path)


# --- Phase 1: launch (detached) ----------------------------------------------
def launch() -> str:
    run_id = secrets.token_hex(3)
    run_dir = ROOT / run_id
    sweep = [
        {'lr': 1e-3, 'steps': 8, 'step_seconds': 0.3},
        {'lr': 1e-2, 'steps': 8, 'step_seconds': 0.3},
        {'lr': 4e-2, 'steps': 8, 'step_seconds': 0.3, 'crash_at': 3},  # one job fails on purpose
    ]
    (run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / 'manifest.json').write_text(json.dumps({'run_id': run_id, 'n_jobs': len(sweep)}))

    for job_id, cfg in enumerate(sweep):
        job_dir = run_dir / 'jobs' / str(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / 'cfg.pkl').write_bytes(pickle.dumps(cfg))
        _merge_status(job_dir, job_id=str(job_id), state='PENDING', step=0, total=cfg['steps'])
        # Detached: start_new_session puts the child in its own process group, so
        # it is NOT killed when this launcher exits. This is the local stand-in
        # for Modal's Function.spawn / app.run(detach=True).
        subprocess.Popen(
            [sys.executable, __file__, 'worker', str(job_dir)],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return run_id


# --- The worker: runs one job, persists progress + result/error --------------
def worker(job_dir_str: str) -> None:
    job_dir = Path(job_dir_str)
    cfg = pickle.loads((job_dir / 'cfg.pkl').read_bytes())
    sink = FileStatusSink(job_dir)
    _merge_status(job_dir, state='RUNNING')
    # Reuse mini's progress context unchanged; only the sink is durable.
    try:
        with progress_context(
            run_id=job_dir.parent.parent.name, job_id=job_dir.name, queue=sink, emission_interval=0.0
        ):
            result = train(cfg)
        (job_dir / 'result.pkl').write_bytes(pickle.dumps(result))
        _merge_status(job_dir, state='DONE')
    except Exception:
        (job_dir / 'error.txt').write_text(traceback.format_exc())
        _merge_status(job_dir, state='FAILED', error=traceback.format_exc().strip().splitlines()[-1])


# --- Phase 2: poll (fresh process reads durable state) -----------------------
def read_statuses(run_id: str) -> list[JobStatus]:
    run_dir = ROOT / run_id
    out: list[JobStatus] = []
    for job_dir in sorted((run_dir / 'jobs').iterdir(), key=lambda p: int(p.name)):
        raw = json.loads((job_dir / 'status.json').read_text())
        out.append(JobStatus(**{k: raw.get(k) for k in JobStatus.__annotations__ if k in raw}))
    return out


def poll(run_id: str) -> None:
    statuses = read_statuses(run_id)
    done = sum(s.state in ('DONE', 'FAILED') for s in statuses)
    print(f'run {run_id}: {done}/{len(statuses)} settled')
    for s in statuses:
        bar = f'{s.step}/{s.total}'
        line = f'  job {s.job_id}: {s.state:8} {bar:>7}  {s.message}'
        if s.error:
            line += f'  !! {s.error}'
        print(line)


# --- Phase 3: gather (fresh process collects results) ------------------------
def gather(run_id: str) -> None:
    run_dir = ROOT / run_id
    print(f'run {run_id} results:')
    for job_dir in sorted((run_dir / 'jobs').iterdir(), key=lambda p: int(p.name)):
        rp = job_dir / 'result.pkl'
        if rp.exists():
            print(f'  job {job_dir.name}: {pickle.loads(rp.read_bytes())}')
        else:
            err = job_dir / 'error.txt'
            tail = err.read_text().strip().splitlines()[-1] if err.exists() else 'no result yet'
            print(f'  job {job_dir.name}: FAILED/incomplete -> {tail}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'launch':
        print(launch())
    elif cmd == 'worker':
        worker(sys.argv[2])
    elif cmd == 'poll':
        poll(sys.argv[2])
    elif cmd == 'gather':
        gather(sys.argv[2])
    else:
        raise SystemExit(f'unknown command: {cmd}')

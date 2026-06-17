"""
Durable run state for detached (agentic) execution.

A `Run` is a handle to work that outlives the process that launched it. The
launcher writes a manifest to a `ControlPlane` and spawns detached workers; later
— possibly in a different process, on a different day — `status()`, `results()`,
`retry()`, and `cancel()` reconstruct everything from durable state.

Two planes (see notes/agentic-experiments.md):
  - control plane: small, hot, last-writer-wins (per-job state, metrics,
    heartbeat). Locally this is JSON under ``<data_dir>/.control``.
  - I/O plane: large artifacts and results. Locally this is the Volume
    (``<data_dir>``); per-job results land under ``_runs/<token>/<job>/``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import cloudpickle

__all__ = ['RunState', 'JobStatus', 'ControlPlane', 'LocalControlPlane', 'Run', 'open_run', 'open_experiment']

DATA_ROOT = Path('.mini')


class RunState(StrEnum):
    PENDING = 'pending'
    RUNNING = 'running'
    DONE = 'done'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


SETTLED = {RunState.DONE, RunState.FAILED, RunState.CANCELLED}


@dataclass
class JobStatus:
    """A point-in-time view of one job, read from the control plane."""

    job_id: str
    state: RunState
    step: int = 0
    total: int = 0
    message: str = ''
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None  # last line of the traceback, if FAILED
    heartbeat_at: float | None = None

    @property
    def settled(self) -> bool:
        return self.state in SETTLED


# ---------------------------------------------------------------------------
# Control plane
# ---------------------------------------------------------------------------


class ControlPlane(ABC):
    """Durable, pollable store of run/job records, addressed by experiment name.

    The local backend (below) is JSON on disk. A Modal backend would put the
    same records in a named ``modal.Dict`` so they're readable from the client
    without a remote function. The interface is deliberately small.
    """

    @abstractmethod
    def create_run(self, run_id: str, spec: dict[str, Any], n_jobs: int) -> None: ...
    @abstractmethod
    def read_spec(self, run_id: str) -> dict[str, Any]: ...
    @abstractmethod
    def read_run(self, run_id: str) -> dict[str, Any]: ...
    @abstractmethod
    def write_run(self, run_id: str, **fields: Any) -> None: ...
    @abstractmethod
    def list_job_ids(self, run_id: str) -> list[str]: ...
    @abstractmethod
    def read_job(self, run_id: str, job_id: str) -> JobStatus: ...
    @abstractmethod
    def write_job(self, run_id: str, job_id: str, **fields: Any) -> None: ...
    @abstractmethod
    def list_runs(self) -> dict[str, dict[str, Any]]: ...


def _atomic_write(path: Path, text: str) -> None:
    """Write via tmp+rename so concurrent readers never see a half-written file."""
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def _merge_json(path: Path, fields: dict[str, Any]) -> None:
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(fields)
    _atomic_write(path, json.dumps(cur))


class LocalControlPlane(ControlPlane):
    """Control plane backed by JSON files under ``<data_dir>/.control``."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _run_dir(self, run_id: str) -> Path:
        return self.root / 'runs' / _token(run_id)

    def _job_path(self, run_id: str, job_id: str) -> Path:
        return self._run_dir(run_id) / 'jobs' / f'{job_id}.json'

    def create_run(self, run_id: str, spec: dict[str, Any], n_jobs: int) -> None:
        run_dir = self._run_dir(run_id)
        (run_dir / 'jobs').mkdir(parents=True, exist_ok=True)
        (run_dir / 'spec.pkl').write_bytes(cloudpickle.dumps(spec))
        now = time.time()
        _atomic_write(
            run_dir / 'run.json',
            json.dumps({'run_id': run_id, 'state': RunState.PENDING, 'n_jobs': n_jobs, 'created_at': now, 'pid': None}),
        )
        for j in range(n_jobs):
            _atomic_write(
                self._job_path(run_id, str(j)),
                json.dumps({'job_id': str(j), 'state': RunState.PENDING, 'step': 0, 'total': 0}),
            )
        self.root.mkdir(parents=True, exist_ok=True)
        _merge_json(
            self.root / 'index.json', {_token(run_id): {'state': RunState.PENDING, 'created_at': now, 'n_jobs': n_jobs}}
        )

    def read_spec(self, run_id: str) -> dict[str, Any]:
        return cloudpickle.loads((self._run_dir(run_id) / 'spec.pkl').read_bytes())

    def read_run(self, run_id: str) -> dict[str, Any]:
        return json.loads((self._run_dir(run_id) / 'run.json').read_text())

    def write_run(self, run_id: str, **fields: Any) -> None:
        _merge_json(self._run_dir(run_id) / 'run.json', fields)
        if 'state' in fields:
            _merge_json(
                self.root / 'index.json', {_token(run_id): {**self._index_entry(run_id), 'state': fields['state']}}
            )

    def _index_entry(self, run_id: str) -> dict[str, Any]:
        idx = self.list_runs()
        return idx.get(_token(run_id), {})

    def list_job_ids(self, run_id: str) -> list[str]:
        jobs_dir = self._run_dir(run_id) / 'jobs'
        names = [p.stem for p in jobs_dir.glob('*.json')]
        return sorted(names, key=lambda n: int(n))

    def read_job(self, run_id: str, job_id: str) -> JobStatus:
        raw = json.loads(self._job_path(run_id, job_id).read_text())
        return JobStatus(
            job_id=raw['job_id'],
            state=RunState(raw['state']),
            step=raw.get('step', 0),
            total=raw.get('total', 0),
            message=raw.get('message', ''),
            metrics=raw.get('metrics', {}),
            error=raw.get('error'),
            heartbeat_at=raw.get('heartbeat_at'),
        )

    def write_job(self, run_id: str, job_id: str, **fields: Any) -> None:
        _merge_json(self._job_path(run_id, job_id), fields)

    def list_runs(self) -> dict[str, dict[str, Any]]:
        path = self.root / 'index.json'
        return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------------------
# Run handle
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """A durable, id-addressable handle to a launched run."""

    id: str
    cp: ControlPlane
    data_dir: Path

    @property
    def token(self) -> str:
        return _token(self.id)

    def status(self) -> list[JobStatus]:
        """Poll the control plane for per-job state. Cheap and stateless."""
        return [self.cp.read_job(self.id, j) for j in self.cp.list_job_ids(self.id)]

    def state(self) -> RunState:
        """Aggregate run state from the jobs."""
        states = [s.state for s in self.status()]
        if not states:
            return RunState.DONE  # no jobs: vacuously done
        if all(s == RunState.DONE for s in states):
            return RunState.DONE
        if any(s == RunState.CANCELLED for s in states) and all(s in SETTLED for s in states):
            return RunState.CANCELLED
        if all(s in SETTLED for s in states):
            return RunState.FAILED
        return RunState.RUNNING

    def done(self) -> bool:
        return self.state() in SETTLED

    def results(self, wait: bool = False, timeout: float = 600.0, poll: float = 0.5) -> list[Any]:
        """Collect per-job results from the I/O plane, in job order.

        Raises if any job is unfinished (unless *wait*) or failed.
        """
        if wait:
            deadline = time.monotonic() + timeout
            while not self.done():
                if time.monotonic() > deadline:
                    raise TimeoutError(f'run {self.id} not settled after {timeout}s')
                time.sleep(poll)
        out: list[Any] = []
        for s in self.status():
            if s.state != RunState.DONE:
                raise RuntimeError(f'job {s.job_id} is {s.state} (not done); cannot gather results')
            out.append(cloudpickle.loads((self._job_dir(s.job_id) / 'result.pkl').read_bytes()))
        return out

    def logs(self, job_id: str) -> str:
        """Return the full traceback / log for a job from the I/O plane."""
        err = self._job_dir(job_id) / 'error.txt'
        return err.read_text() if err.exists() else '(no logs)'

    def retry(self, failed_only: bool = True) -> Run:
        """Reset unfinished/failed jobs to PENDING and relaunch a worker."""
        targets = (
            {RunState.FAILED} if failed_only else (SETTLED - {RunState.DONE}) | {RunState.PENDING, RunState.RUNNING}
        )
        for s in self.status():
            if s.state in targets and s.state != RunState.DONE:
                self.cp.write_job(self.id, s.job_id, state=RunState.PENDING, error=None)
        self.cp.write_run(self.id, state=RunState.PENDING)
        spawn_worker(self.data_dir, self.id)
        return self

    def cancel(self) -> None:
        """Signal the detached worker to stop and mark unfinished jobs CANCELLED."""
        pid = self.cp.read_run(self.id).get('pid')
        if pid:
            with _suppress_os_error():
                os.killpg(pid, signal.SIGTERM)  # worker is a session leader: pgid == pid
        for s in self.status():
            if not s.settled:
                self.cp.write_job(self.id, s.job_id, state=RunState.CANCELLED)
        self.cp.write_run(self.id, state=RunState.CANCELLED)

    def _job_dir(self, job_id: str) -> Path:
        return self.data_dir / '_runs' / self.token / job_id


# ---------------------------------------------------------------------------
# Discovery & worker spawning
# ---------------------------------------------------------------------------


def open_run(run_id: str, data_root: Path | str = DATA_ROOT) -> Run:
    """Reconstruct a `Run` from its id in a fresh process."""
    data_dir = Path(data_root) / _experiment(run_id)
    return _run_at(data_dir, run_id)


def open_experiment(name: str, data_root: Path | str = DATA_ROOT) -> tuple[LocalControlPlane, list[str]]:
    """Return the control plane and run tokens (newest first) for an experiment.

    The key payoff of name-addressing: we can discover runs we never launched,
    without knowing any run id.
    """
    cp = LocalControlPlane(Path(data_root) / name / '.control')
    runs = cp.list_runs()
    tokens = sorted(runs, key=lambda t: runs[t].get('created_at', 0), reverse=True)
    return cp, tokens


def latest_run(name: str, data_root: Path | str = DATA_ROOT) -> Run | None:
    cp, tokens = open_experiment(name, data_root)
    return open_run(f'{name}/{tokens[0]}', data_root) if tokens else None


def spawn_worker(data_dir: Path, run_id: str) -> int:
    """Launch a detached worker for *run_id*; return its pid (== its pgid)."""
    proc = subprocess.Popen(
        [sys.executable, '-m', 'mini._worker', str(data_dir), run_id],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.pid


def _run_at(data_dir: Path, run_id: str) -> Run:
    return Run(run_id, LocalControlPlane(Path(data_dir) / '.control'), Path(data_dir))


def _experiment(run_id: str) -> str:
    return run_id.split('/', 1)[0]


def _token(run_id: str) -> str:
    return run_id.split('/', 1)[1] if '/' in run_id else run_id


class _suppress_os_error:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        return exc_type is not None and issubclass(exc_type, (ProcessLookupError, PermissionError))

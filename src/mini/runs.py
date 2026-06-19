"""
Shared durable-state primitives for the memoized orchestration.

The control plane is small, hot, last-writer-wins JSON (per-task state, metrics,
heartbeat); the I/O plane holds the large artifacts. This module owns the bits
both planes and both backends need: the ``RunState`` enum, atomic/merge JSON
writes (so concurrent readers never see a half-written file), and the detached
task-worker spawn. The rest of the state model lives in ``mini.memo``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from enum import StrEnum
from pathlib import Path

__all__ = ['RunState', 'SETTLED', 'data_root', 'spawn_taskworker']

# Markers that identify a project root, in priority order.
_ROOT_MARKERS = ('pyproject.toml', '.git')


def data_root() -> Path:
    """The project's ``.mini`` store, anchored at the project root.

    Every ``mini`` command shares one store regardless of cwd: we walk up from the
    current directory for a project marker (``pyproject.toml`` / ``.git``) and put
    ``.mini`` beside it, falling back to cwd if none is found. Resolved *lazily*
    (per call, off the live cwd) — not frozen at import — so a process that changes
    directory, and tests that ``chdir`` into a tmp dir, both see the right root.
    The path is absolute, so detached workers stay correct under their own cwd.
    """
    cwd = Path.cwd().resolve()
    for d in (cwd, *cwd.parents):
        if any((d / m).exists() for m in _ROOT_MARKERS):
            return d / '.mini'
    return cwd / '.mini'


class RunState(StrEnum):
    PENDING = 'pending'
    RUNNING = 'running'
    DONE = 'done'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


SETTLED = {RunState.DONE, RunState.FAILED, RunState.CANCELLED}


def _atomic_write(path: Path, text: str) -> None:
    """Write via tmp+rename so concurrent readers never see a half-written file."""
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def _merge_json(path: Path, fields: dict) -> None:
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(fields)
    _atomic_write(path, json.dumps(cur))


def spawn_taskworker(data_dir: Path, key: str) -> int:
    """Launch a detached worker for one memoized task *key*; return its pid.

    The local implementation of ``Apparatus.spawn_task``: a subprocess that runs
    the staged call (``MemoStore._call``) and persists its result/state under the
    content key, outliving the orchestration tick that launched it.
    """
    proc = subprocess.Popen(
        [sys.executable, '-m', 'mini._taskworker', str(data_dir), key],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.pid

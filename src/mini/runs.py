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
import platform
import subprocess
import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "RunState",
    "SETTLED",
    "STALE_HEARTBEAT_S",
    "compute_env",
    "data_root",
    "is_queued",
    "spawn_taskworker",
    "stale_heartbeat",
]

# Markers that identify a project root, in priority order.
_ROOT_MARKERS = ("pyproject.toml", ".git")


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
            return d / ".mini"
    return cwd / ".mini"


def _gpus() -> tuple[str | None, int]:
    """Best-effort GPU model + count, dependency-free. NVIDIA exposes a per-GPU
    info file on Linux; we don't import torch/jax just to name the card.
    """
    model, count = None, 0
    for info in sorted(Path("/proc/driver/nvidia/gpus").glob("*/information")):
        count += 1
        if model is not None:
            continue
        try:
            for line in info.read_text().splitlines():
                if line.startswith("Model:"):
                    model = line.split(":", 1)[1].strip()
        except OSError:
            continue
    return model, count


def _mem_total_gb() -> float | None:
    """Total RAM visible to the process, in GiB (from ``/proc/meminfo``), or ``None``."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)  # kB -> GiB
    except OSError, ValueError, IndexError:
        pass
    return None


# Modal stamps these into every container's env. Safe to record (workspace/region/
# container ids — the forensic breadcrumb a run needs). Deliberately *excludes*
# MODAL_IDENTITY_TOKEN / MODAL_TASK_SECRET / MODAL_TOKEN_* — those are credentials
# and must never enter a record.
_MODAL_ENV_KEYS = {
    "MODAL_TASK_ID": "modal_task_id",
    "MODAL_ENVIRONMENT": "modal_environment",
    "MODAL_REGION": "region",
    "MODAL_CLOUD_PROVIDER": "cloud",
    "MODAL_IMAGE_ID": "modal_image_id",
}


def compute_env() -> dict[str, Any]:
    """A snapshot of *what a task actually ran on*, recorded by the worker.

    Captured inside the worker process (local subprocess or Modal container), so it
    reflects the real execution environment rather than the requested backend —
    useful when a sweep fans out across heterogeneous Modal containers. Kept small
    (it rides the hot control-plane record): host, OS/arch, Python, CPU/RAM, the GPU
    model + count if any, and — on Modal — the container/region/cloud ids (never any
    token or secret; see ``_MODAL_ENV_KEYS``).
    """
    env: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if (mem := _mem_total_gb()) is not None:
        env["mem_total_gb"] = mem
    gpu, gpu_count = _gpus()
    if gpu:
        env["gpu"] = gpu
    if gpu_count > 1:
        env["gpu_count"] = gpu_count
    for src, dst in _MODAL_ENV_KEYS.items():
        if val := os.environ.get(src):
            env[dst] = val
    return env


class RunState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


SETTLED = {RunState.DONE, RunState.FAILED, RunState.CANCELLED}


def is_queued(rec: dict) -> bool:
    """Launched but no worker has started yet — queued, not actually running.

    The client claims RUNNING *before* the apparatus spawns the worker, and
    ``env`` is the worker's first write once it truly starts. So RUNNING with no
    ``env`` means the task is still waiting to be scheduled: a momentary blip
    locally, but on Modal a capacity-starved task can sit here indefinitely
    (only the wall-clock budget reaps it). Display-only — settling stays with
    ``reap_dead``/``enforce_budget``.
    """
    return rec.get("state") == RunState.RUNNING and not rec.get("env")


STALE_HEARTBEAT_S = 300.0
"""Heartbeat age past which a RUNNING task is *advisorily* flagged stale.

Heartbeats ride on progress emissions (there is no fixed cadence), so staleness
is a hint, not proof: a worker deep in a non-emitting stretch (a heavy import,
one long step) looks the same as a dead one. Five minutes is comfortably past
any healthy emission gap we've seen while still catching zombies early.
"""


def stale_heartbeat(rec: dict, now: float | None = None) -> bool:
    """Is this RUNNING task's heartbeat suspiciously old — worker possibly dead?

    Backend-agnostic and *display-only*: badges in ``status``/``watch`` use it to
    keep the human/agent-visible signal honest even where a backend liveness
    probe has a blind spot (#20). Settling stays with ``reap_dead``/
    ``enforce_budget``; a queued record's heartbeat is just its launch stamp, so
    queued tasks are never stale (they get the ``⧖`` treatment instead).
    """
    if rec.get("state") != RunState.RUNNING or is_queued(rec):
        return False
    hb = rec.get("heartbeat_at")
    return bool(hb) and ((now if now is not None else time.time()) - hb) > STALE_HEARTBEAT_S


def _atomic_write(path: Path, text: str) -> None:
    """Write via tmp+rename so concurrent readers never see a half-written file."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
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
        [sys.executable, "-m", "mini._taskworker", str(data_dir), key],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.pid

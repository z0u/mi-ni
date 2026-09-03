"""
Shared durable-state primitives for the memoized orchestration.

The control plane is small, hot, last-writer-wins JSON (per-task state, metrics, heartbeat); the I/O plane holds the large artifacts. This module owns the bits both planes and both backends need: the ``RunState`` enum, atomic/merge JSON writes (so concurrent readers never see a half-written file), and the detached task-worker spawn. The rest of the state model lives in ``mini.memo``.
"""

from __future__ import annotations

import functools
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "RunState",
    "SETTLED",
    "STALE_HEARTBEAT_S",
    "compute_env",
    "data_root",
    "describe_numerics_drift",
    "in_declared_phase",
    "installed_numerics",
    "is_queued",
    "merged_numerics_drift",
    "numerics_drift",
    "progress_age",
    "spawn_taskworker",
    "stale_heartbeat",
    "stale_progress",
]

# Markers that identify a project root, in priority order.
_ROOT_MARKERS = ("pyproject.toml", ".git")


def data_root() -> Path:
    """The project's ``.mini`` store, anchored at the project root.

    Every ``mini`` command shares one store regardless of cwd: we walk up from the current directory for a project marker (``pyproject.toml`` / ``.git``) and put ``.mini`` beside it, falling back to cwd if none is found. Resolved *lazily* (per call, off the live cwd) — not frozen at import — so a process that changes directory, and tests that ``chdir`` into a tmp dir, both see the right root. The path is absolute, so detached workers stay correct under their own cwd.
    """
    cwd = Path.cwd().resolve()
    for d in (cwd, *cwd.parents):
        if any((d / m).exists() for m in _ROOT_MARKERS):
            return d / ".mini"
    return cwd / ".mini"


def _gpus() -> tuple[str | None, int]:
    """Best-effort GPU model + count, dependency-free. NVIDIA exposes a per-GPU info file on Linux; we don't import torch/jax just to name the card."""
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

# Env that changes what a task *computes*, as opposed to where it runs. Recorded
# verbatim so a number can be traced back to the numerics that produced it —
# `XLA_FLAGS` decides whether a GPU reduction is deterministic, so two attempts
# that agree on code and inputs can still disagree on results across a change
# here. Read from the worker's own environment rather than from what the client
# asked for, so a setting that failed to arrive shows up as absent.
_NUMERICS_ENV_KEYS = ("XLA_FLAGS",)

# Packages whose *version* changes what a task computes. They are invisible to the
# memo — `mini.memo._is_project_file` excludes site-packages, so no library source
# reaches a fingerprint, and no version string reaches a key — yet a jax upgrade
# measurably moves a training result (jax 0.10.1 → 0.11.0 shifted a fixed nGPT
# run's final loss by ~1 part in 4×10⁶; `eng/determinism.md`). Recorded so a number
# can be traced back to the library that produced it, and so a memo hit computed
# under a since-upgraded library can be *noticed* (`numerics_drift`). Read in the
# worker, like the flags above, so what's recorded is what actually ran.
#
# Kept to the ones a measurement has implicated. Adding a name here only affects
# records written afterwards: an older record simply doesn't carry it, and reads
# as quiet rather than as drifted.
_NUMERICS_PACKAGES = ("jax", "jaxlib", "numpy")


@functools.cache
def installed_numerics() -> dict[str, str]:
    """Installed versions of the numerics packages (``_NUMERICS_PACKAGES``), by name.

    Cached: a process's own installed set can't change under it, and this is read once per memo hit on the driver's hot path. Packages that aren't installed are simply absent — the driver of an experiment that never touches jax says nothing about jax.
    """
    versions = {}
    for name in _NUMERICS_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def numerics_drift(
    env: Mapping[str, Any] | None, current: Mapping[str, str] | None = None
) -> dict[str, tuple[str, str]]:
    """Numerics packages that have moved since *env* was recorded: ``{name: (recorded, current)}``.

    Detection only, and deliberately not evidence: folding versions into the memo key would invalidate every record in the project on an upgrade, to reproduce numbers we already have (`eng/determinism.md`). What this answers instead is the question a memo hit can't ask for itself — "was this result computed under the libraries I have now?" — so a run that straddles an upgrade can be said out loud rather than found later.

    *current* defaults to this process's own installed set, which on Modal is the driver's rather than the container's; the image is frozen from the same lock, so they agree unless something has been pinned apart deliberately. A package missing from either side is skipped: a record written before the version was captured, or a driver without the package installed, is an absence of evidence rather than evidence of a change.
    """
    recorded = (env or {}).get("numerics_packages") or {}
    now = installed_numerics() if current is None else current
    return {name: (was, now[name]) for name, was in recorded.items() if name in now and now[name] != was}


def _version_sort_key(v: str) -> tuple[tuple[int, int | str], ...]:
    """Numeric-aware sort key for version strings, without a parser dependency: ``0.9.0`` sorts before ``0.10.1``, and non-numeric fragments compare as text."""
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in re.split(r"(\d+)", v))


def merged_numerics_drift(drifts: Iterable[Mapping[str, tuple[str, str]]]) -> dict[str, tuple[tuple[str, ...], str]]:
    """Fold per-record drifts into one map: ``{package: (recorded versions, current)}``.

    A run can straddle *several* upgrades — half a sweep computed under jax 0.9.0, half under 0.10.1 — so a package's recorded side is a tuple of every version seen, sorted, rather than whichever record happened to be read last. The current side is single-valued: it's this process's installed version.
    """
    recorded: dict[str, set[str]] = {}
    current: dict[str, str] = {}
    for drift in drifts:
        for name, (was, now) in drift.items():
            recorded.setdefault(name, set()).add(was)
            current[name] = now
    return {name: (tuple(sorted(vs, key=_version_sort_key)), current[name]) for name, vs in recorded.items()}


def describe_numerics_drift(moved: Mapping[str, tuple[tuple[str, ...], str]]) -> str:
    """One clause per package: ``jax 0.9.0/0.10.1 → 0.11.0, numpy 2.2.3 → 2.3.0``."""
    return ", ".join(f"{name} {'/'.join(was)} → {now}" for name, (was, now) in sorted(moved.items()))


def compute_env() -> dict[str, Any]:
    """A snapshot of *what a task actually ran on*, recorded by the worker.

    Captured inside the worker process (local subprocess or Modal container), so it reflects the real execution environment rather than the requested backend — useful when a sweep fans out across heterogeneous Modal containers. Kept small (it rides the hot control-plane record): host, OS/arch, Python, CPU/RAM, the GPU model + count if any, the numerics env (``_NUMERICS_ENV_KEYS``) and numerics package versions (``_NUMERICS_PACKAGES``), and — on Modal — the container/region/cloud ids (never any token or secret; see ``_MODAL_ENV_KEYS``).
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
    if numerics := {k: v for k in _NUMERICS_ENV_KEYS if (v := os.environ.get(k))}:
        env["numerics_env"] = numerics
    if packages := installed_numerics():
        env["numerics_packages"] = dict(packages)
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

    The client claims RUNNING *before* the apparatus spawns the worker, and ``env`` is the worker's first write once it truly starts. So RUNNING with no ``env`` means the task is still waiting to be scheduled: a momentary blip locally, but on Modal a capacity-starved task can sit here indefinitely (only the wall-clock budget reaps it). Display-only — settling stays with ``reap_dead``/``enforce_budget``.
    """
    return rec.get("state") == RunState.RUNNING and not rec.get("env")


STALE_HEARTBEAT_S = 300.0
"""Heartbeat age past which a RUNNING task is *advisorily* flagged stale.

Heartbeats ride on progress emissions (there is no fixed cadence), so staleness
is a hint, not proof: a worker deep in a non-emitting stretch (a heavy import,
one long step) looks the same as a dead one. Five minutes is comfortably past
any healthy emission gap we've seen while still catching zombies early.
"""


def in_declared_phase(rec: dict, now: float | None = None) -> bool:
    """Is this record inside a step-free span the task declared and still has budget for?

    The record side of :func:`~mini.progress.blocking_phase`: the worker stamps ``phase`` (a label) and ``phase_until`` (the deadline of the widest span still open) on entry, and clears them at the last exit. Both liveness badges consult this, because a declared span is the one case where the quiet is the work. It reads the *deadline* rather than the label, so the span bounds the silence instead of exempting it — once a budget lapses, the badges resume, and a worker that died mid-upload is flagged as soon as its own allowance runs out.
    """
    until = rec.get("phase_until")
    return bool(until) and (now if now is not None else time.time()) < until


def stale_heartbeat(rec: dict, now: float | None = None) -> bool:
    """Is this RUNNING task's heartbeat suspiciously old — worker possibly dead?

    Backend-agnostic and *display-only*: badges in ``status``/``watch`` use it to keep the human/agent-visible signal honest even where a backend liveness probe has a blind spot (#20). Settling stays with ``reap_dead``/ ``enforce_budget``; a queued record's heartbeat is just its launch stamp, so queued tasks are never stale (they get the ``⧖`` treatment instead).

    A task inside a declared blocking phase (:func:`in_declared_phase`) is never stale. Heartbeats ride on progress emissions, so a task whose whole body is store transfers — a publish step downloading its inputs and uploading one result — emits nothing for its entire run, and its heartbeat sits at ``started_at`` while everything is healthy. The span is what tells the two apart.
    """
    if rec.get("state") != RunState.RUNNING or is_queued(rec):
        return False
    now = now if now is not None else time.time()
    if in_declared_phase(rec, now):
        return False
    hb = rec.get("heartbeat_at")
    return bool(hb) and (now - hb) > STALE_HEARTBEAT_S


def progress_age(rec: dict, now: float | None = None) -> float | None:
    """Seconds since this RUNNING task's step last advanced, or ``None``.

    Anchored on ``progress_at`` (the worker's last ``(step, total)`` advance) and falling back to ``started_at`` for a worker that hasn't emitted yet — so the age is meaningful from the moment the worker truly starts. ``None`` for queued/settled records, where "progress" has no referent.
    """
    if rec.get("state") != RunState.RUNNING or is_queued(rec):
        return None
    anchor = rec.get("progress_at") or rec.get("started_at")
    return ((now if now is not None else time.time()) - anchor) if anchor else None


def stale_progress(rec: dict, now: float | None = None) -> bool:
    """Is this RUNNING task's *step* frozen suspiciously long — worker possibly wedged?

    The companion to :func:`stale_heartbeat` for the wedge failure mode: a hung device call or deadlocked thread can leave emissions (heartbeats) flowing while ``step`` never advances, so heartbeat staleness never trips. Display-only, like the heartbeat badge. The threshold is the worker's own watchdog threshold when it stamped one (past it, the watchdog itself has evidently failed to fire — worth flagging loudly), else the generic ``STALE_HEARTBEAT_S``. Before the first emission the watchdog applies its startup grace instead of the tight timeout, so the badge matches: a worker legitimately tokenizing for ten minutes is not "possibly wedged" yet.

    A task inside a declared blocking phase (:func:`in_declared_phase` — a checkpoint upload, a large pull) is not stale until that budget runs out. Same reasoning as the grace: the span has no steps to report and is healthy anyway.

    A *recently closed* span counts too, via ``phase_at``. A task whose body is a run of consecutive transfers holds no deadline in the moments between them, so without this a poll landing in a gap sees a step frozen since ``started_at`` and calls a healthy worker wedged. Crossing a span boundary is evidence the task moved — not step-shaped, but movement — so it is measured against the same threshold as a step. It buys no more blindness than a step emission would: a task that closes a span and *then* wedges still trips once the threshold passes.
    """
    age = progress_age(rec, now)
    if age is None:
        return False
    now = now if now is not None else time.time()
    if in_declared_phase(rec, now):
        return False
    if rec.get("progress_at"):
        threshold = rec.get("watchdog_s") or STALE_HEARTBEAT_S
    else:  # still in setup: no emission yet, so the (looser) grace governs
        threshold = rec.get("watchdog_grace_s") or rec.get("watchdog_s") or STALE_HEARTBEAT_S
    if (at := rec.get("phase_at")) and (now - at) <= threshold:
        return False
    return age > threshold


def _atomic_write(path: Path, text: str) -> None:
    """Write via tmp+rename so concurrent readers never see a half-written file."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _merge_json(path: Path, fields: dict) -> None:
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(fields)
    _atomic_write(path, json.dumps(cur))


def spawn_taskworker(data_dir: Path, key: str, env: dict[str, str] | None = None) -> int:
    """Launch a detached worker for one memoized task *key*; return its pid.

    The local implementation of ``Apparatus.spawn_task``: a subprocess that runs the staged call (``MemoStore._call``) and persists its result/state under the content key, outliving the orchestration tick that launched it.

    *env* is overlaid on the parent environment, the local counterpart of Modal's per-container env: a fresh process per task, so a library that reads its env once at init sees this task's setting rather than an earlier task's.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "mini._taskworker", str(data_dir), key],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=(os.environ | env) if env else None,
    )
    return proc.pid

"""
Content-addressed memoization for multi-step orchestration.

The cache key must be **deterministic across processes** (every agent wake is a
fresh process) and should **invalidate when the relevant code changes** so the
"fix a bug, re-run" loop works. Hashing ``cloudpickle.dumps(fn)`` fails the first
test — its bytes vary run to run. So we fingerprint the function's *source*, plus
the source of the project functions/classes it references (transitively), which
is stable and captures dependencies in your own code while ignoring library
churn (site-packages and the mini framework itself are excluded).
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import time
import types
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import cloudpickle

from mini.runs import SETTLED, RunState, _atomic_write, _merge_json

__all__ = ['fingerprint', 'RecordStore', 'LocalRecordStore', 'MemoStore', 'PollCache', 'META_KEY']

# Source under these roots is treated as an opaque, stable dependency: the
# stdlib, installed packages, and the mini framework itself (so editing mini
# doesn't invalidate every experiment's cache).
_MINI_DIR = str(Path(__file__).parent.resolve())

# Reserved control-plane key for run-level metadata (the wall-clock budget /
# deadline). It rides the same record store as the task records — a sidecar, so a
# detached run carries its budget with no new infra — but is excluded from
# ``records()`` so it never reads as a task or skews the aggregate state. A task
# fingerprint is ``{name}-{hex12}``, so ``__run__`` can never collide with one.
META_KEY = '__run__'


def _is_project_source(obj: Any) -> bool:
    try:
        f = inspect.getsourcefile(obj)
    except TypeError, OSError:
        return False
    if not f:
        return False
    rf = str(Path(f).resolve())
    return 'site-packages' not in rf and '/lib/python3' not in rf and not rf.startswith(_MINI_DIR)


def _collect_sources(fn: Callable, seen: dict[str, str]) -> None:
    qualname = getattr(fn, '__qualname__', repr(fn))
    if qualname in seen:
        return
    try:
        seen[qualname] = inspect.getsource(fn)
    except TypeError, OSError:
        return
    code = getattr(fn, '__code__', None)
    g = getattr(fn, '__globals__', {})
    refs: list[Any] = [g[n] for n in code.co_names if n in g] if code is not None else []
    for cell in getattr(fn, '__closure__', None) or []:
        try:
            refs.append(cell.cell_contents)
        except ValueError:
            pass
    for obj in refs:
        if isinstance(obj, types.FunctionType) and _is_project_source(obj):
            _collect_sources(obj, seen)
        elif isinstance(obj, type) and _is_project_source(obj):
            try:
                seen.setdefault(obj.__qualname__, inspect.getsource(obj))
            except TypeError, OSError:
                pass


def _code_fingerprint(fn: Callable) -> str:
    seen: dict[str, str] = {}
    _collect_sources(fn, seen)
    blob = '\n--\n'.join(f'{k}:{v}' for k, v in sorted(seen.items()))
    return hashlib.sha256(blob.encode()).hexdigest()


def _canonical(o: Any) -> Any:
    """Normalize *o* into a JSON-stable structure — deterministic across processes.

    ``pickle.dumps`` is *not* stable run-to-run for values containing sets, and a
    Pydantic model carries one (``__pydantic_fields_set__``); set iteration order
    is hash-randomized per process, so the same config would fingerprint
    differently each wake and miss the cache (the same trap that ruled out
    cloudpickle for the *code* fingerprint). So we canonicalize first: models and
    dataclasses to their field dicts, sets to sorted lists, then JSON with sorted
    keys downstream.
    """
    dump = getattr(o, 'model_dump', None)  # pydantic v2, duck-typed (no hard dep on pydantic)
    if callable(dump):
        try:
            return _canonical(dump(mode='json'))
        except TypeError:
            return _canonical(dump())
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _canonical(dataclasses.asdict(o))
    if isinstance(o, Mapping):
        return {str(k): _canonical(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return sorted((_canonical(x) for x in o), key=lambda v: json.dumps(v, sort_keys=True, default=repr))
    if isinstance(o, (list, tuple)):
        return [_canonical(x) for x in o]
    return o


def _input_fingerprint(args: tuple) -> str:
    try:
        blob = json.dumps(_canonical(args), sort_keys=True, default=repr).encode()
    except Exception:
        blob = repr(args).encode()
    return hashlib.sha256(blob).hexdigest()


def fingerprint(fn: Callable, args: tuple, version: str | None = None) -> str:
    """A stable content key for calling *fn* with *args* (the memo key)."""
    h = hashlib.sha256()
    h.update(_code_fingerprint(fn).encode())
    h.update(_input_fingerprint(args).encode())
    if version:
        h.update(version.encode())
    return f'{getattr(fn, "__name__", "task")}-{h.hexdigest()[:12]}'


class RecordStore(ABC):
    """A small, flat ``key -> record`` store: the memo's control plane.

    Records are tiny and hot (state, step, latest metrics, heartbeat),
    last-writer-wins. The local backend is JSON files; the Modal backend is a
    named ``modal.Dict`` (readable from the client with no remote function). The
    interface is deliberately minimal so a ``modal.Dict`` satisfies it directly.
    """

    @abstractmethod
    def read(self, key: str) -> dict[str, Any] | None: ...
    @abstractmethod
    def write(self, key: str, record: dict[str, Any]) -> None:
        """Overwrite a record wholesale (resets stale fields, e.g. a prior error)."""

    @abstractmethod
    def merge(self, key: str, fields: dict[str, Any]) -> None:
        """Merge *fields* into the record (progress/heartbeat updates)."""

    @abstractmethod
    def keys(self) -> list[str]: ...


class LocalRecordStore(RecordStore):
    """``RecordStore`` backed by JSON files under a directory."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def read(self, key: str) -> dict[str, Any] | None:
        p = self.root / f'{key}.json'
        return json.loads(p.read_text()) if p.exists() else None

    def write(self, key: str, record: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.root / f'{key}.json', json.dumps(record))

    def merge(self, key: str, fields: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _merge_json(self.root / f'{key}.json', fields)

    def keys(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob('*.json')) if self.root.exists() else []


class MemoStore:
    """Per-experiment content-addressed task store (the orchestration backend).

    Two planes: records (small: state,
    metrics, heartbeat) live on a ``RecordStore`` control plane; results and
    tracebacks (large) live on the I/O plane. Locally both are files under
    ``data_dir``; on Modal the records go to a ``modal.Dict`` and results to the
    Volume, so the same ``MemoStore`` serves the client (poll/gather) and the
    remote worker (write-back) without either touching the other's filesystem.

    The cloudpickled *call* is not part of either plane: locally it's staged to
    disk for the subprocess worker; on Modal it's passed straight to ``spawn``.
    """

    def __init__(self, data_dir: Path, records: RecordStore | None = None):
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / '.control' / 'memo'
        self.records_backend: RecordStore = records or LocalRecordStore(self.root)

    def _call(self, key: str) -> Path:
        return self.root / f'{key}.pkl'

    def result_dir(self, key: str) -> Path:
        return self.data_dir / '_memo' / key

    def state(self, key: str) -> RunState | None:
        rec = self.records_backend.read(key)
        return RunState(rec['state']) if rec and rec.get('state') else None

    def record(self, key: str) -> dict[str, Any]:
        return self.records_backend.read(key) or {'key': key, 'state': None}

    def result(self, key: str) -> Any:
        return cloudpickle.loads((self.result_dir(key) / 'result.pkl').read_bytes())

    def error(self, key: str) -> str:
        e = self.result_dir(key) / 'error.txt'
        return e.read_text() if e.exists() else '(no logs)'

    def update(self, key: str, **fields: Any) -> None:
        self.records_backend.merge(key, fields)

    def records(self) -> list[dict[str, Any]]:
        return [
            rec for key in self.records_backend.keys() if key != META_KEY and (rec := self.records_backend.read(key))
        ]

    def meta(self) -> dict[str, Any]:
        """Run-level metadata (the wall-clock budget / ``deadline_at``), or ``{}``.

        Stored under the reserved ``META_KEY`` so it shares the run's control plane
        (local JSON / Modal ``Dict``) without ever surfacing as a task.
        """
        return self.records_backend.read(META_KEY) or {}

    def set_meta(self, **fields: Any) -> None:
        """Merge run-level metadata (e.g. ``deadline_at``) into the reserved record."""
        self.records_backend.merge(META_KEY, fields)

    def deadline(self) -> float | None:
        """The run's wall-clock deadline (epoch seconds), or ``None`` if unbudgeted."""
        return self.meta().get('deadline_at')

    def budget_expired(self) -> bool:
        """Whether a budget is set *and* its deadline has passed.

        The gate both for tearing a run down (cancel in-flight tasks) and for
        refusing to launch new work past the deadline.
        """
        d = self.deadline()
        return d is not None and time.time() >= d

    def reset(self, key: str) -> None:
        """Clear a record back to un-run (state → None) so the next tick reruns it.

        The retry primitive: a settled-but-not-DONE task is terminal, so re-running
        takes intent. Stale result/error artifacts are overwritten on the rerun.
        """
        self.records_backend.write(key, {'key': key, 'state': None})

    def mark_running(self, fn: Callable, key: str) -> None:
        """Flip the record to RUNNING (wholesale, clearing any prior error).

        Called by ``Ctx`` before the apparatus spawns the worker, so a poll
        between stage and first heartbeat sees RUNNING rather than a stale state.
        """
        self.records_backend.write(
            key,
            {'key': key, 'fn': getattr(fn, '__name__', 'task'), 'state': RunState.RUNNING, 'created_at': time.time()},
        )

    def write_call(self, key: str, fn: Callable, args: tuple, hooks: list[Callable] | None = None) -> None:
        """Stage the cloudpickled call to disk for a local subprocess worker."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._call(key).write_bytes(cloudpickle.dumps((fn, args, hooks or [])))

    def read_call(self, key: str) -> tuple[Callable, tuple, list[Callable]]:
        return cloudpickle.loads(self._call(key).read_bytes())


class PollCache:
    """Cheap repeated polling of a ``MemoStore``'s records for large sweeps.

    A settled record (``DONE``/``FAILED``/``CANCELLED``) is immutable, so once
    seen it never needs re-reading. Each ``records`` call re-reads only the
    unsettled subset (plus any keys not seen yet); the settled tail is served
    from memory. On Modal every record read is a ``modal.Dict`` round-trip, so a
    long sweep that's mostly done stops paying for the part that can't change —
    the watch loops poll just the handful still in flight.

    A reaper may settle a stale ``RUNNING`` record out from under us. That key was
    unsettled (so not cached), and the reaper writes it through ``MemoStore``, so
    the next ``records`` re-reads it once and caches the now-terminal record —
    nothing stale lingers.
    """

    def __init__(self) -> None:
        self._settled: dict[str, dict[str, Any]] = {}

    def records(self, store: MemoStore) -> list[dict[str, Any]]:
        backend = store.records_backend
        out: list[dict[str, Any]] = []
        for key in backend.keys():
            if key == META_KEY:  # run-level metadata, not a task
                continue
            if cached := self._settled.get(key):
                out.append(cached)
                continue
            if (rec := backend.read(key)) is None:
                continue
            if rec.get('state') in SETTLED:  # StrEnum members hash as their str value
                self._settled[key] = rec
            out.append(rec)
        return out

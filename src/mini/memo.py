"""
Identity-keyed memoization for multi-step orchestration.

A task record answers two different questions, and the store keeps them apart:

- **Identity — which task is this?** The key: the fn's qualified name plus a
  fingerprint of its inputs. Stable across code edits, so a record (and its
  logs, results, history) keeps one address for the task's whole life.
- **Validity — is the cached result current?** The *evidence* stored on each
  attempt: a fingerprint of the fn's source plus the source of the project
  functions/classes it references (transitively), and the explicit ``version=``.
  Stale evidence re-runs the task **in place** — a new attempt under the same
  key, with the prior attempt compacted into the record's ``history``.

Both fingerprints must be **deterministic across processes** (every agent wake
is a fresh process) — hashing ``cloudpickle.dumps(fn)`` fails that (its bytes
vary run to run), so we fingerprint *source*, which also ignores library churn
(site-packages and the mini framework itself are excluded).
"""

from __future__ import annotations

import dataclasses
import dis
import enum
import fcntl
import functools
import hashlib
import inspect
import json
import logging
import secrets
import time
import types
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PurePath
from typing import Any, Callable, Iterator

import cloudpickle

from mini.runs import SETTLED, RunState, _atomic_write, _merge_json

__all__ = ["task_key", "task_key_parts", "RecordStore", "LocalRecordStore", "MemoStore", "PollCache", "META_KEY"]

log = logging.getLogger(__name__)

# Source under these roots is treated as an opaque, stable dependency: the
# stdlib, installed packages, and the mini framework itself (so editing mini
# doesn't invalidate every experiment's cache).
_MINI_DIR = str(Path(__file__).parent.resolve())

# Reserved control-plane key for run-level metadata (the wall-clock budget /
# deadline). It rides the same record store as the task records — a sidecar, so a
# detached run carries its budget with no new infra — but is excluded from
# ``records()`` so it never reads as a task or skews the aggregate state. A task
# fingerprint is ``{name}-{hex12}``, so ``__run__`` can never collide with one.
META_KEY = "__run__"


def _is_project_source(obj: Any) -> bool:
    try:
        f = inspect.getsourcefile(obj)
    except TypeError, OSError:
        return False
    if not f:
        return False
    rf = str(Path(f).resolve())
    return "site-packages" not in rf and "/lib/python3" not in rf and not rf.startswith(_MINI_DIR)


def _nested_codes(code: types.CodeType) -> Iterator[types.CodeType]:
    """*code* plus every code object nested in it (inner defs, lambdas, genexprs).

    A helper referenced only inside a nested function lives in the *inner* code
    object's ``co_names``; walking just the outer one would miss it.
    """
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _nested_codes(const)


def _attr_chain_refs(fn: Callable) -> list[Any]:
    """Objects reached through attribute chains rooted at a global (``utils.helper``).

    Bare names are resolved via ``co_names`` ∩ globals, but a helper called as a
    module attribute never appears in globals under its own name — so without this
    walk, ``import utils; utils.helper()`` would be *invisible* to the fingerprint
    and editing the helper would silently serve stale results. We scan the bytecode
    for ``LOAD_GLOBAL`` → ``LOAD_ATTR``… chains and resolve each step with
    ``getattr``, collecting any function/class the chain lands on (``pkg.mod.fn``
    resolves through the intermediate modules).
    """
    code = getattr(fn, "__code__", None)
    g = getattr(fn, "__globals__", {})
    if code is None:
        return []
    refs: list[Any] = []
    for c in _nested_codes(code):
        chain: Any = None
        for ins in dis.get_instructions(c):
            if ins.opname == "LOAD_GLOBAL" and ins.argval in g:
                chain = g[ins.argval]
            elif ins.opname == "LOAD_ATTR" and chain is not None:
                chain = getattr(chain, ins.argval, None)
                if callable(chain) or isinstance(chain, type):
                    refs.append(chain)
            else:
                chain = None  # any other instruction breaks the chain
    return refs


def _collect_class(cls: type, seen: dict[str, str]) -> None:
    """Collect a class's source, then traverse its methods' *references*.

    The class source already contains the method bodies textually (so editing a
    method invalidates); traversing the methods is what picks up the helpers and
    project bases they *call*, which the text alone doesn't reach.
    """
    if cls.__qualname__ in seen:
        return
    try:
        seen[cls.__qualname__] = inspect.getsource(cls)
    except TypeError, OSError:
        return
    for base in cls.__bases__:
        if _is_project_source(base):
            _collect_class(base, seen)
    for member in vars(cls).values():
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if isinstance(member, types.FunctionType):
            _collect_sources(member, seen)


def _value_json(obj: Any) -> str | None:
    """A stable JSON encoding of a plain value, or ``None`` if it has none.

    No ``default=`` fallback here: an exotic object's ``repr`` can embed a memory
    address, which would make the fingerprint differ every process — worse than
    not tracking the value at all. Stable-or-skip.
    """
    try:
        return json.dumps(_canonical(obj), sort_keys=True)
    except TypeError, ValueError:
        return None


def _named_refs(fn: Callable) -> list[tuple[str | None, Any]]:
    """Everything *fn* references, as ``(name, object)`` pairs.

    Bare globals (from every nested code object), closure cells (named via
    ``co_freevars``), and attribute-chain targets (unnamed — they're never
    treated as values, only as code).
    """
    code = getattr(fn, "__code__", None)
    g = getattr(fn, "__globals__", {})
    names = [n for c in _nested_codes(code) for n in c.co_names] if code is not None else []
    refs: list[tuple[str | None, Any]] = [(n, g[n]) for n in names if n in g]
    freevars = code.co_freevars if code is not None else ()
    for name, cell in zip(freevars, getattr(fn, "__closure__", None) or (), strict=False):
        try:
            refs.append((name, cell.cell_contents))
        except ValueError:
            pass
    return refs + [(None, obj) for obj in _attr_chain_refs(fn)]


def _collect_sources(fn: Callable, seen: dict[str, str]) -> None:
    qualname = getattr(fn, "__qualname__", repr(fn))
    if qualname in seen:
        return
    try:
        seen[qualname] = inspect.getsource(fn)
    except TypeError, OSError:
        return
    for name, obj in _named_refs(fn):
        if isinstance(obj, types.MethodType):
            obj = obj.__func__
        if isinstance(obj, types.FunctionType) and _is_project_source(obj):
            _collect_sources(obj, seen)
        elif isinstance(obj, type) and _is_project_source(obj):
            _collect_class(obj, seen)
        elif name is not None and not isinstance(obj, types.ModuleType) and not callable(obj):
            # A plain value referenced by name (a module-level LR, a config table):
            # fold its canonical JSON in, so editing the *value* invalidates like
            # editing code. Skipped when it has no stable encoding (see _value_json).
            if (js := _value_json(obj)) is not None:
                seen[f"{qualname}::{name}"] = js


@functools.lru_cache(maxsize=256)
def _sources_for(fn: Callable) -> tuple[tuple[str, str], ...]:
    """The (cached) sorted dependency manifest for *fn*: ``(name, source-or-value)``.

    Source never changes within one process (every wake is a fresh process), so
    this caches per fn object — a ``ctx.map`` fingerprints its fn once per wake
    instead of re-walking the reference graph for every cell.
    """
    seen: dict[str, str] = {}
    _collect_sources(fn, seen)
    return tuple(sorted(seen.items()))


def _code_fingerprint(fn: Callable) -> str:
    blob = "\n--\n".join(f"{k}:{v}" for k, v in _manifest(fn))
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
    dump = getattr(o, "model_dump", None)  # pydantic v2, duck-typed (no hard dep on pydantic)
    if callable(dump):
        try:
            return _canonical(dump(mode="json"))
        except TypeError:
            return _canonical(dump())
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _canonical(dataclasses.asdict(o))
    if isinstance(o, enum.Enum):
        return [type(o).__qualname__, _canonical(o.value)]
    if isinstance(o, Mapping):
        return {str(k): _canonical(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return sorted((_canonical(x) for x in o), key=lambda v: json.dumps(v, sort_keys=True, default=repr))
    if isinstance(o, (list, tuple)):
        return [_canonical(x) for x in o]
    if isinstance(o, PurePath):
        return str(o)  # a path keys by *location* — prefer an Artifact, which keys by content
    if (code := _canonical_code(o)) is not None:
        return code
    return o


def _canonical_code(o: Any) -> list | None:
    """Canonical forms for *code passed as data* — keyed by source, not identity.

    A callable's repr embeds a memory address, so without these a callable input
    would produce a fresh key every process — relaunching the task on every wake.
    Returns ``None`` for non-code values (handled by :func:`_canonical`).
    """
    if isinstance(o, types.MethodType):
        return ["method", o.__func__.__qualname__, _code_fingerprint(o.__func__)[:12], _canonical(o.__self__)]
    if isinstance(o, types.FunctionType):
        return ["fn", o.__qualname__, _code_fingerprint(o)[:12]]
    if isinstance(o, type):
        return ["class", o.__qualname__, hashlib.sha256(_class_source(o).encode()).hexdigest()[:12]]
    if isinstance(o, functools.partial):
        return ["partial", _canonical(o.func), _canonical(o.args), _canonical(o.keywords)]
    return None


def _class_source(cls: type) -> str:
    try:
        return inspect.getsource(cls)
    except TypeError, OSError:
        return cls.__qualname__  # no source (builtin/C) — the name is the stable id


def _input_fingerprint(args: tuple) -> str:
    try:
        blob = json.dumps(_canonical(args), sort_keys=True, default=repr).encode()
    except Exception:
        blob = repr(args).encode()
    if b" at 0x" in blob:  # an object address leaked into the key
        log.warning(
            "task inputs have no stable encoding (repr contains an object address), so the memo "
            "key will differ every process and the task can never be a cache hit — it will relaunch "
            "on every wake. Pass plain data, dataclasses, or Artifact handles instead: %.200r",
            args,
        )
    return hashlib.sha256(blob).hexdigest()


# Guards fn-value cycles: a module-level container holding the fn that references
# it would recurse (collect fn → canonicalize the container → fingerprint fn → …).
# The marker is a constant per qualname, so the resulting manifest stays
# deterministic across processes.
_collecting: set[int] = set()


def _manifest(fn: Callable) -> tuple[tuple[str, str], ...]:
    if id(fn) in _collecting:
        return ((f"<recursive:{getattr(fn, '__qualname__', '?')}>", ""),)
    _collecting.add(id(fn))
    try:
        try:
            return _sources_for(fn)
        except TypeError:  # unhashable callable — compute uncached
            seen: dict[str, str] = {}
            _collect_sources(fn, seen)
            return tuple(sorted(seen.items()))
    finally:
        _collecting.discard(id(fn))


def task_key(fn: Callable, args: tuple) -> str:
    """The stable *identity* key for calling *fn* with *args*.

    Identity is which task this is — the fn's qualified name plus its inputs —
    deliberately excluding code: an edited fn re-runs under the **same** key (a
    new attempt on the same record) instead of orphaning it. Whether the cached
    result is still *valid* is judged against the evidence from
    :func:`task_key_parts`, stored per attempt.
    """
    return task_key_parts(fn, args)[0]


def task_key_parts(fn: Callable, args: tuple, version: str | None = None) -> tuple[str, dict[str, Any]]:
    """The identity key plus the validity evidence to stamp on its next attempt.

    Returns ``(key, parts)``. The key hashes only the fn's module-qualified name
    and the input fingerprint. *parts* carries what decides staleness — the code
    fingerprint, ``version=``, and ``deps``: a short hash per tracked dependency
    (the fn itself, each project helper/class it references, each plain-value
    global) — so ``mini explain`` can diff two attempts down to *which*
    dependency moved.
    """
    manifest = _manifest(fn)
    blob = "\n--\n".join(f"{k}:{v}" for k, v in manifest)
    code_fp = hashlib.sha256(blob.encode()).hexdigest()
    input_fp = _input_fingerprint(args)
    ident = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', 'task')}"
    h = hashlib.sha256(f"{ident}\n{input_fp}".encode())
    key = f"{getattr(fn, '__name__', 'task')}-{h.hexdigest()[:12]}"
    deps = {k: hashlib.sha256(v.encode()).hexdigest()[:8] for k, v in manifest}
    parts = {"code_fp": code_fp[:12], "input_fp": input_fp[:12], "deps": deps}
    if version:
        parts["version"] = version
    return key, parts


# What a finished attempt is worth keeping once a new one replaces it: the
# evidence and the outcome. Live/bulky fields (metrics, env, heartbeats, pids)
# describe a worker, not the attempt's identity in the run's story.
_ATTEMPT_KEEP = ("state", "gen", "code_fp", "input_fp", "version", "deps", "created_at", "error", "exc_type")


def _compact_attempt(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: rec[k] for k in _ATTEMPT_KEEP if k in rec}


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

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a record entirely (the GC verb) — a no-op if *key* is absent."""

    # Conditional writes, fenced on the record's attempt generation (``gen``).
    # These defaults are read-check-write — atomic only if the backend makes them
    # so (``LocalRecordStore`` overrides under a file lock; ``modal.Dict`` has no
    # compare-and-swap, so on Modal only the fresh-key claim is exact — via
    # insert-if-absent, see ``ModalRecordStore.write_if`` — and the rest is
    # best-effort with a tiny window — still a vast improvement over
    # unconditional last-writer-wins).

    def write_if(self, key: str, record: dict[str, Any], gen: str | None) -> bool:
        """Replace the record iff its current ``gen`` equals *gen* (``None`` = unclaimed)."""
        if (self.read(key) or {}).get("gen") != gen:
            return False
        self.write(key, record)
        return True

    def merge_if(self, key: str, fields: dict[str, Any], gen: str | None) -> bool:
        """Merge *fields* iff the record's current ``gen`` equals *gen*."""
        if (self.read(key) or {}).get("gen") != gen:
            return False
        self.merge(key, fields)
        return True


class LocalRecordStore(RecordStore):
    """``RecordStore`` backed by JSON files under a directory.

    All mutations serialize on one store-wide ``flock``: ``merge`` is
    read-modify-write, so without the lock two concurrent mergers (a worker's
    final DONE vs the reaper's FAILED, a heartbeat vs the tick's pid stamp)
    could each read the same base record and silently drop the other's fields.
    Reads stay lock-free — ``_atomic_write`` renames, so a reader never sees a
    half-written file. The lock also makes ``write_if``/``merge_if`` genuinely
    atomic (check and write under one critical section).
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / ".lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield  # released when the file closes

    def read(self, key: str) -> dict[str, Any] | None:
        p = self.root / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def write(self, key: str, record: dict[str, Any]) -> None:
        with self._locked():
            _atomic_write(self.root / f"{key}.json", json.dumps(record))

    def merge(self, key: str, fields: dict[str, Any]) -> None:
        with self._locked():
            _merge_json(self.root / f"{key}.json", fields)

    def write_if(self, key: str, record: dict[str, Any], gen: str | None) -> bool:
        with self._locked():
            if (self.read(key) or {}).get("gen") != gen:
                return False
            _atomic_write(self.root / f"{key}.json", json.dumps(record))
            return True

    def merge_if(self, key: str, fields: dict[str, Any], gen: str | None) -> bool:
        with self._locked():
            if (self.read(key) or {}).get("gen") != gen:
                return False
            _merge_json(self.root / f"{key}.json", fields)
            return True

    def keys(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json")) if self.root.exists() else []

    def delete(self, key: str) -> None:
        with self._locked():
            (self.root / f"{key}.json").unlink(missing_ok=True)


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
        self.root = self.data_dir / ".control" / "memo"
        self.records_backend: RecordStore = records or LocalRecordStore(self.root)

    def _call(self, key: str) -> Path:
        return self.root / f"{key}.pkl"

    def result_dir(self, key: str) -> Path:
        return self.data_dir / "_memo" / key

    def state(self, key: str) -> RunState | None:
        rec = self.records_backend.read(key)
        return RunState(rec["state"]) if rec and rec.get("state") else None

    def record(self, key: str) -> dict[str, Any]:
        return self.records_backend.read(key) or {"key": key, "state": None}

    def result_path(self, key: str, gen: str | None) -> Path:
        """Where attempt *gen* of *key* writes its result.

        Generation-qualified so a superseded worker that survives ``cancel``
        physically *cannot* overwrite its successor's result — each attempt owns
        its own file, and readers resolve through the record's current ``gen``.
        (``None`` — a record from before generations — reads the legacy name.)
        """
        return self.result_dir(key) / (f"result-{gen}.pkl" if gen else "result.pkl")

    def error_path(self, key: str, gen: str | None) -> Path:
        return self.result_dir(key) / (f"error-{gen}.txt" if gen else "error.txt")

    def artifacts_path(self, key: str, gen: str | None) -> Path:
        """Where attempt *gen* records the blob shas its result references.

        The worker stamps this sidecar next to the result (see
        :func:`mini._taskworker.execute_task`), so the artifact GC can mark a
        result's references without unpickling it — no project imports, no
        arbitrary code, one small read per record however large the result.
        """
        return self.result_dir(key) / (f"result-{gen}.artifacts.json" if gen else "result.artifacts.json")

    def result_artifacts(self, key: str) -> list[str] | None:
        """Blob shas the current result references, or ``None`` for a record
        from before the sidecar existed (unpickle the result to find out).
        """
        p = self.artifacts_path(key, self._gen(key))
        return json.loads(p.read_text()) if p.exists() else None

    def _gen(self, key: str) -> str | None:
        return (self.records_backend.read(key) or {}).get("gen")

    def result(self, key: str) -> Any:
        return cloudpickle.loads(self.result_path(key, self._gen(key)).read_bytes())

    def error(self, key: str) -> str:
        for p in (self.error_path(key, self._gen(key)), self.error_path(key, None)):
            if p.exists():
                return p.read_text()
        return "(no logs)"

    def update(self, key: str, **fields: Any) -> None:
        self.records_backend.merge(key, fields)

    def update_if(self, key: str, gen: str, **fields: Any) -> bool:
        """Merge *fields* only while attempt *gen* still owns the record.

        The worker-side fence: every write a worker makes passes through here, so
        once its record is claimed by a successor attempt (or released by
        ``cancel``/``reap_dead``), a lingering worker can no longer heartbeat, merge
        DONE over the new attempt's RUNNING, or resurrect cleared fields.
        """
        return self.records_backend.merge_if(key, fields, gen)

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

    def requested_keys(self) -> list[str] | None:
        """The keys the DAG requested on its last tick, or ``None`` if never recorded.

        Records are content-keyed, so an edited fn or a removed config leaves its old
        record behind under a key no wake will request again. This manifest is what
        lets a read-only view (``status``/``ls``/``watch``) aggregate over the run's
        *current* records and mark the rest superseded — without re-running ``main``
        (reads must never tick). ``None`` (a store written before the manifest, or a
        run never ticked) means "unknown": treat every record as current.
        """
        return self.meta().get("requested")

    def split_current(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split *records* into ``(current, superseded)`` against the requested set.

        With no manifest, everything is current (nothing to judge against).
        """
        requested = self.requested_keys()
        if requested is None:
            return records, []
        wanted = set(requested)
        current = [r for r in records if r["key"] in wanted]
        return current, [r for r in records if r["key"] not in wanted]

    def deadline(self) -> float | None:
        """The run's wall-clock deadline (epoch seconds), or ``None`` if unbudgeted."""
        return self.meta().get("deadline_at")

    def budget_expired(self) -> bool:
        """Whether a budget is set *and* its deadline has passed.

        The gate both for tearing a run down (cancel in-flight tasks) and for
        refusing to launch new work past the deadline.
        """
        d = self.deadline()
        return d is not None and time.time() >= d

    def _with_history(self, key: str, rec: dict[str, Any]) -> dict[str, Any]:
        """Fold the record's prior attempt (if any ran) into *rec*'s ``history``.

        Keys are identity, so a re-run replaces the record in place; compacting
        the outgoing attempt first is what keeps the task's story — every attempt,
        its evidence, its outcome — on the one record (``mini explain``).
        """
        prior = self.records_backend.read(key) or {}
        history: list[dict[str, Any]] = list(prior.get("history") or ())
        if prior.get("state"):  # a reset placeholder (state None) is not an attempt
            history.append(_compact_attempt(prior))
        if history:
            rec["history"] = history
        return rec

    def reset(self, key: str) -> None:
        """Clear a record back to un-run (state → None) so the next tick reruns it.

        The retry primitive: a settled-but-not-DONE task is terminal, so re-running
        takes intent. The cleared attempt is kept in the record's history; stale
        result/error artifacts are overwritten on the rerun.
        """
        self.records_backend.write(key, self._with_history(key, {"key": key, "state": None}))

    def mark_running(
        self, fn: Callable, key: str, parts: dict[str, Any] | None = None, expect_gen: str | None = None
    ) -> str | None:
        """Claim the record for a fresh attempt: flip it to RUNNING (wholesale,
        clearing any prior error) under a new generation stamp.

        Called by ``Ctx`` before the apparatus spawns the worker, so a poll
        between stage and first heartbeat sees RUNNING rather than a stale state.
        *parts* is the validity evidence (``code_fp``/``version``/``deps`` from
        :func:`task_key_parts`) this attempt runs under; any prior attempt is
        compacted into ``history``, so ``mini explain`` can answer "why did this
        re-run" after the fact.

        The claim is conditional on *expect_gen* — the ``gen`` the caller read
        when it classified the record (``None`` = unclaimed). If another ticker
        claimed the key in between, nothing is written and ``None`` returns
        (don't spawn — theirs is running). On success, returns the new attempt's
        ``gen``: the fence every write from that worker must carry.
        """
        gen = secrets.token_hex(4)
        rec = self._with_history(
            key,
            {
                "key": key,
                "fn": getattr(fn, "__name__", "task"),
                "state": RunState.RUNNING,
                "gen": gen,
                "created_at": time.time(),
                **(parts or {}),
            },
        )
        return gen if self.records_backend.write_if(key, rec, expect_gen) else None

    def write_call(
        self, key: str, fn: Callable, args: tuple, hooks: list[Callable] | None = None, gen: str | None = None
    ) -> None:
        """Stage the cloudpickled call to disk for a local subprocess worker."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._call(key).write_bytes(cloudpickle.dumps((fn, args, hooks or [], gen)))

    def read_call(self, key: str) -> tuple[Callable, tuple, list[Callable], str | None]:
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

    A *tick* can relaunch a settled record in place (keys are identity; an edit
    makes a new attempt, it doesn't re-key), so a cache must not outlive a tick —
    ``drive_and_watch`` rebuilds its cache per stage. Between ticks, settled is
    settled.
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
            if rec.get("state") in SETTLED:  # StrEnum members hash as their str value
                self._settled[key] = rec
            out.append(rec)
        return out

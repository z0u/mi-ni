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

import hashlib
import inspect
import json
import pickle
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import cloudpickle

from mini.runs import RunState, _atomic_write, _merge_json

__all__ = ['fingerprint', 'MemoStore']

# Source under these roots is treated as an opaque, stable dependency: the
# stdlib, installed packages, and the mini framework itself (so editing mini
# doesn't invalidate every experiment's cache).
_MINI_DIR = str(Path(__file__).parent.resolve())


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


def _input_fingerprint(args: tuple) -> str:
    try:
        blob = pickle.dumps(args, protocol=4)  # stable for dict/list/tuple/str/num
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


class MemoStore:
    """Per-experiment content-addressed task store (the orchestration backend).

    Records (small: state, metrics, heartbeat) live on the control plane; results
    and tracebacks (large) live on the I/O plane — the same two-plane split as
    runs. Each task is launched as its own detached worker, keyed by content.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / '.control' / 'memo'

    def _rec(self, key: str) -> Path:
        return self.root / f'{key}.json'

    def _call(self, key: str) -> Path:
        return self.root / f'{key}.pkl'

    def result_dir(self, key: str) -> Path:
        return self.data_dir / '_memo' / key

    def state(self, key: str) -> RunState | None:
        p = self._rec(key)
        return RunState(json.loads(p.read_text())['state']) if p.exists() else None

    def record(self, key: str) -> dict[str, Any]:
        p = self._rec(key)
        return json.loads(p.read_text()) if p.exists() else {'key': key, 'state': None}

    def result(self, key: str) -> Any:
        return cloudpickle.loads((self.result_dir(key) / 'result.pkl').read_bytes())

    def error(self, key: str) -> str:
        e = self.result_dir(key) / 'error.txt'
        return e.read_text() if e.exists() else '(no logs)'

    def update(self, key: str, **fields: Any) -> None:
        _merge_json(self._rec(key), fields)

    def records(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        return [json.loads(p.read_text()) for p in sorted(self.root.glob('*.json'))]

    def launch(self, fn: Callable, args: tuple, key: str, hooks: list[Callable] | None = None) -> None:
        """Persist the call and spawn a detached worker for it."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._call(key).write_bytes(cloudpickle.dumps((fn, args, hooks or [])))
        _atomic_write(
            self._rec(key),
            json.dumps(
                {
                    'key': key,
                    'fn': getattr(fn, '__name__', 'task'),
                    'state': RunState.RUNNING,
                    'created_at': time.time(),
                }
            ),
        )
        subprocess.Popen(
            [sys.executable, '-m', 'mini._taskworker', str(self.data_dir), key],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

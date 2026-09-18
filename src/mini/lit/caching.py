"""
A cache for the expensive calls a document makes: figures, fits, anything slow.

``@memo`` keys a call the way :mod:`mini.memo` keys a task — the function's qualified name plus a fingerprint of its inputs is the identity, and a fingerprint of its source (and the project code it references, transitively) is the validity evidence — so editing the plot function re-renders the figure, and editing the prose around it does not. The value is pickled under ``.mini/lit-cache/``; a hit is served from memory within a process and from disk across processes, which is what makes a fresh ``render`` of a figure-heavy report take well under a second.

A memoized function that writes assets through the current :class:`~mini.reports.Publisher` (a ``themed`` figure writes two PNGs) has those files recorded with its value, and the hit is honoured only while they exist — so clearing the output directory re-draws, and a stale cache can never point at a missing image.

Inputs need a stable encoding. Plain data, dataclasses, and NumPy arrays are handled (an array is hashed by its bytes); an object whose ``repr`` carries a memory address makes the call miss every time, and :mod:`mini.memo` logs a warning when that happens.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar, overload

from mini.memo import task_key_parts
from mini.reports import current_publisher

__all__ = ["memo", "cache_dir", "set_cache_dir"]

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_cache_dir: Path | None = None
_hot: dict[tuple[str, str], Any] = {}


def cache_dir() -> Path:
    if _cache_dir is None:
        from mini.runs import data_root

        return data_root() / "lit-cache"
    return _cache_dir


def set_cache_dir(path: Path | str | None) -> None:
    global _cache_dir
    _cache_dir = None if path is None else Path(path)
    _hot.clear()


def _prepare(o: Any) -> Any:
    """Replace the inputs :func:`mini.memo.task_key_parts` cannot encode stably with digests it can."""
    mod = type(o).__module__
    if mod.startswith("numpy") and hasattr(o, "tobytes"):
        return ["ndarray", str(o.dtype), list(o.shape), hashlib.sha256(o.tobytes()).hexdigest()[:16]]
    if mod.startswith("pandas") and hasattr(o, "to_numpy"):
        import pandas as pd

        h = pd.util.hash_pandas_object(o, index=True).to_numpy()
        return ["pandas", type(o).__name__, hashlib.sha256(h.tobytes()).hexdigest()[:16]]
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {"__dataclass__": type(o).__qualname__} | {
            f.name: _prepare(getattr(o, f.name)) for f in dataclasses.fields(o)
        }
    if isinstance(o, dict):
        return {str(k): _prepare(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_prepare(x) for x in o]
    return o


@overload
def memo(fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def memo(*, version: str | None = ...) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def memo(fn: Callable[P, R] | None = None, /, *, version: str | None = None) -> Any:
    """Memoize *fn* on disk, keyed by its source and inputs (see the module docstring).

    ``version=`` is an explicit invalidation lever for a change the source fingerprint cannot see (new data under an unchanged ref name).
    """

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key, parts = task_key_parts(fn, (_prepare(args), _prepare(kwargs)), version)
            evidence = f"{parts['code_fp']}:{parts.get('version', '')}"
            pub = current_publisher()
            asset_dir = pub.asset_dir if pub is not None else None

            if (hit := _hot.get((key, evidence))) is not None and _assets_present(hit["assets"], asset_dir):
                return hit["value"]
            path = cache_dir() / f"{key}.pkl"
            if path.exists():
                try:
                    rec = pickle.loads(path.read_bytes())
                except Exception:
                    rec = None
                if rec and rec.get("evidence") == evidence and _assets_present(rec["assets"], asset_dir):
                    _hot[key, evidence] = rec
                    return rec["value"]

            mark = len(pub.log) if pub is not None else 0
            value = fn(*args, **kwargs)
            assets = pub.log[mark:] if pub is not None else []
            rec = {"evidence": evidence, "value": value, "assets": assets, "deps": parts["deps"]}
            _hot[key, evidence] = rec
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp")  # per process: two renders may cache the same key at once
                tmp.write_bytes(pickle.dumps(rec))
                tmp.replace(path)
            except Exception as e:  # an unpicklable value still returns; it just isn't cached across processes
                log.warning("lit.caching: %s not cached to disk: %s", getattr(fn, "__qualname__", fn), e)
            return value

        return wrapper

    return decorate(fn) if fn is not None else decorate


def _assets_present(names: list[str], asset_dir: Path | None) -> bool:
    if not names:
        return True
    return asset_dir is not None and all((asset_dir / n).exists() for n in names)

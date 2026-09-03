"""
mi-ni: infrastructure for running memoized experiments.

The convenience names below are re-exported *lazily* (PEP 562). Binding them at
import time would make every ``import mini.anything`` execute the whole package
first — ``modal``, ``asyncio``, ``rich``, ``grpclib`` — because Python runs a
package's ``__init__`` before any of its submodules. That put a ~0.3s toll and a
full environment behind a leaf module like :mod:`mini.reports`, which itself
imports nothing but the standard library.

``__getattr__`` imports the owning submodule on first touch and caches the value
in the module namespace, so later lookups skip this path entirely.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Static-only: the same names the lazy map serves, so type-checkers and IDEs
    # resolve `from mini import Ctx` without the runtime cost.
    from mini.apparatus import Apparatus
    from mini.experiment import Experiment, load_experiment
    from mini.local_apparatus import LocalApparatus
    from mini.modal_apparatus import ModalApparatus
    from mini.orchestration import MISSING, Ctx, MemoError, Pending, TaskFailed, tick
    from mini.progress import ProgressMessage, blocking_phase, emit_metrics, emit_progress
    from mini.runs import RunState
    from mini.store import Artifact, LocalStore, StaleWriteError, Store, store_context
    from mini.volume import get_data_dir

#: Which submodule owns each re-exported name. Grouped to mirror the import block
#: above, then flattened for lookup. `__all__` restates the names as a literal
#: because ruff and `ty` only read the literal form; `test_lazy_exports.py` holds
#: all three in step.
_EXPORTS: dict[str, str] = {
    name: module
    for module, names in {
        "mini.apparatus": ["Apparatus"],
        "mini.experiment": ["Experiment", "load_experiment"],
        "mini.local_apparatus": ["LocalApparatus"],
        "mini.modal_apparatus": ["ModalApparatus"],
        "mini.orchestration": ["MISSING", "Ctx", "MemoError", "Pending", "TaskFailed", "tick"],
        "mini.progress": ["ProgressMessage", "blocking_phase", "emit_metrics", "emit_progress"],
        "mini.runs": ["RunState"],
        "mini.store": ["Artifact", "LocalStore", "StaleWriteError", "Store", "store_context"],
        "mini.volume": ["get_data_dir"],
    }.items()
    for name in names
}

__all__ = [
    "Apparatus",
    "LocalApparatus",
    "ModalApparatus",
    "ProgressMessage",
    "emit_progress",
    "emit_metrics",
    "blocking_phase",
    "get_data_dir",
    "Artifact",
    "StaleWriteError",
    "Store",
    "LocalStore",
    "store_context",
    "Experiment",
    "load_experiment",
    "RunState",
    "Ctx",
    "MemoError",
    "Pending",
    "TaskFailed",
    "MISSING",
    "tick",
]


def __getattr__(name: str) -> Any:
    """Import the submodule that owns `name`, then cache the value here (PEP 562)."""
    if (module := _EXPORTS.get(name)) is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

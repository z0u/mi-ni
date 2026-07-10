"""
Importable experiment definitions.

An experiment is a ``main(ctx)`` orchestration. It carries no notebook/UI state,
so the CLI and detached workers can both import it; the notebook becomes a report
that reads durable results.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:
    from mini.apparatus import Apparatus
    from mini.orchestration import Ctx

__all__ = ["Experiment", "load_experiment"]


@dataclass
class Experiment:
    """A named experiment with a memoized multi-step orchestration.

    The definition carries no compute: the apparatus is injected at execution
    (by the CLI or a notebook) — ``tick(exp, apparatus)`` — so the same module
    runs locally or remotely without edits::

        def main(ctx):
            meta = ctx.run(prepare_data, role='prep')    # CPU prep
            return ctx.map(train, [...], role='train')   # per-step GPU

        Experiment(name='pipeline', main=main, roles={'train': dict(gpu='L4')})

    Roles let ``main`` stay backend-agnostic: it names a label and the
    ``roles`` table maps that label to concrete hardware via ``.w()``. A
    table written for Modal still loads locally (local ``.w()`` ignores kwargs).
    """

    name: str
    main: Callable[[Ctx], Any]
    roles: Mapping[str, dict[str, Any]] | Callable[[Apparatus], Mapping[str, Apparatus]] | None = None

    def resolve_roles(self, base: Apparatus) -> dict[str, Apparatus]:
        """Bind role labels to concrete apparatus variants of *base*.

        A dict maps each label to ``.w()`` kwargs applied to *base* (the common
        case); a callable receives *base* and returns the variants directly (for
        per-role ``before_each`` / image / volume). ``None`` → no roles.
        """
        if self.roles is None:
            return {}
        if isinstance(self.roles, Mapping):
            table = cast("Mapping[str, Mapping[str, Any]]", self.roles)
            return {label: base.w(**kwargs) for label, kwargs in table.items()}
        return dict(self.roles(base))


def load_experiment(path: str | Path) -> Experiment:
    """Import a file and return its module-level ``experiment = Experiment(...)``."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"mini_experiment_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experiment from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered only during exec: class creation (e.g. dataclasses on 3.14) needs the
    # module resolvable by name, but leaving it registered would make cloudpickle treat
    # the module as importable and serialize task fns by reference — which a remote
    # worker can't resolve.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    experiment = getattr(module, "experiment", None)
    if not isinstance(experiment, Experiment):
        raise AttributeError(f"{path} must define a module-level `experiment = Experiment(...)`")
    return experiment

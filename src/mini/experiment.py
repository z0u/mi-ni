"""
Importable experiment definitions.

An experiment is either a single sweep (``fn`` + ``configs``) or a multi-step
orchestration (``main``) — see notes/agentic-experiments.md. It carries no
notebook/UI state, so the CLI and the detached workers can both import it; the
notebook becomes a *report* that reads durable results.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from mini.orchestration import Ctx

__all__ = ['Experiment', 'load_experiment']


@dataclass
class Experiment:
    """A named experiment: a single sweep, or a multi-step orchestration.

    The definition carries *no* compute: the apparatus is injected at execution
    (by the CLI or a notebook) — ``tick(exp, apparatus)`` — so the same module
    runs locally or remotely without edits.

    Single sweep (lowered to a one-line map)::

        Experiment(name='sweep', fn=train, configs=[(1e-3,), (1e-2,)])

    Multi-step (memoized; re-run to recover)::

        def main(ctx):
            meta = ctx.run(prepare_data)                 # CPU prep
            return ctx.map(train, [...], on=gpu)         # per-step compute

        Experiment(name='pipeline', main=main)
    """

    name: str
    fn: Callable[..., Any] | None = None
    configs: Sequence[Any] | None = None
    main: Callable[[Ctx], Any] | None = None

    def orchestration(self) -> Callable[[Ctx], Any]:
        """Return the ``main(ctx)`` DAG (single sweeps become a one-line map)."""
        if self.main is not None:
            return self.main
        if self.fn is None or self.configs is None:
            raise ValueError('Experiment needs either main=, or both fn= and configs=')
        fn = self.fn
        items = [c if isinstance(c, tuple) else (c,) for c in self.configs]
        return lambda ctx: ctx.map(fn, items)


def load_experiment(path: str | Path) -> Experiment:
    """Import a file and return its module-level ``experiment = Experiment(...)``."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f'mini_experiment_{path.stem}', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load experiment from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    experiment = getattr(module, 'experiment', None)
    if not isinstance(experiment, Experiment):
        raise AttributeError(f'{path} must define a module-level `experiment = Experiment(...)`')
    return experiment

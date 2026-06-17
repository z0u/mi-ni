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

from mini.apparatus import Apparatus
from mini.local_apparatus import LocalApparatus
from mini.runs import Run

if TYPE_CHECKING:
    from mini.orchestration import Ctx

__all__ = ['Experiment', 'load_experiment']


@dataclass
class Experiment:
    """A named experiment: a single sweep, or a multi-step orchestration.

    Single sweep::

        Experiment(name='sweep', fn=train, configs=[(1e-3,), (1e-2,)])

    Multi-step (memoized; re-run to recover)::

        def main(ctx):
            meta = ctx.run(prepare_data)
            return ctx.map(train, [(lr, meta['vocab']) for lr in lrs])

        Experiment(name='pipeline', main=main)
    """

    name: str
    fn: Callable[..., Any] | None = None
    configs: Sequence[Any] | None = None
    main: Callable[[Ctx], Any] | None = None
    apparatus: Apparatus | None = None

    def make_apparatus(self) -> Apparatus:
        return self.apparatus or LocalApparatus(self.name)

    def data_dir(self) -> Path:
        return self.make_apparatus().volume.path

    def before_hooks(self) -> list[Callable[[], Any]]:
        return list(getattr(self.make_apparatus(), '_before_hooks', []))

    def columns(self) -> list[list[Any]]:
        """Transpose per-job configs into per-argument columns for ``submit``."""
        rows = [c if isinstance(c, tuple) else (c,) for c in (self.configs or [])]
        return [list(col) for col in zip(*rows, strict=False)] if rows else []

    def orchestration(self) -> Callable[[Ctx], Any]:
        """Return the ``main(ctx)`` DAG (single sweeps become a one-line map)."""
        if self.main is not None:
            return self.main
        if self.fn is None or self.configs is None:
            raise ValueError('Experiment needs either main=, or both fn= and configs=')
        fn = self.fn
        items = [c if isinstance(c, tuple) else (c,) for c in self.configs]
        return lambda ctx: ctx.map(fn, items)

    def submit(self) -> Run:
        """Launch a single sweep detached (one-shot run/job model)."""
        if self.fn is None or self.configs is None:
            raise ValueError('submit() needs fn= and configs=; use the orchestration driver for main=')
        return self.make_apparatus().submit(self.fn, *self.columns())


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

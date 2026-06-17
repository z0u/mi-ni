"""
Importable experiment definitions for detached runs.

An experiment is a job function plus the configs to sweep it over (and
optionally which apparatus to use). It carries no notebook/UI state, so the CLI
and the detached workers can both import it. The notebook becomes a *report*
that reads durable results — see notes/agentic-experiments.md.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from mini.apparatus import Apparatus
from mini.local_apparatus import LocalApparatus
from mini.runs import Run

__all__ = ['Experiment', 'load_experiment']


@dataclass
class Experiment:
    """A named, importable sweep definition.

    ``configs`` is one entry per job: a tuple is unpacked as positional args to
    ``fn``; a bare value is passed as the single argument.
    """

    name: str
    fn: Callable[..., Any]
    configs: Sequence[Any]
    apparatus: Apparatus | None = None

    def make_apparatus(self) -> Apparatus:
        return self.apparatus or LocalApparatus(self.name)

    def columns(self) -> list[list[Any]]:
        """Transpose per-job configs into per-argument columns for ``submit``."""
        rows = [c if isinstance(c, tuple) else (c,) for c in self.configs]
        return [list(col) for col in zip(*rows, strict=False)] if rows else []

    def submit(self) -> Run:
        """Launch the sweep detached and return a durable handle."""
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

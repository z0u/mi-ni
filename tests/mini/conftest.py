"""Fixtures for the gc tests, which all need a real completed run to collect from.

`mini gc` reads memo records, result sidecars and staged calls off disk, so nothing
short of driving a local sweep to completion produces the state under test.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest

from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus
from mini.orchestration import tick


def _sweep(name: str, fn, xs: list) -> Experiment:
    return Experiment(name=name, main=lambda ctx: ctx.map(fn, xs))


def _drive(exp: Experiment, app: LocalApparatus, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, _ = tick(exp, app)
        if done:
            return
        time.sleep(0.1)
    raise AssertionError("orchestration did not complete")


@pytest.fixture
def sweep() -> Callable[..., Experiment]:
    """Build an experiment that maps `fn` over `xs` — one memo record per input."""
    return _sweep


@pytest.fixture
def drive() -> Callable[..., None]:
    """Tick an experiment to completion, so it leaves a complete manifest behind."""
    return _drive


@pytest.fixture
def with_superseded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[str], LocalApparatus]:
    """A completed run with one superseded record: sweep [1, 2, 3], then drop 3."""

    def build(name: str) -> LocalApparatus:
        monkeypatch.chdir(tmp_path)

        def train(x):
            return x * 2

        app = LocalApparatus(name)
        _drive(_sweep(name, train, [1, 2, 3]), app)
        _drive(_sweep(name, train, [1, 2]), app)  # config 3 removed → its record superseded
        return app

    return build

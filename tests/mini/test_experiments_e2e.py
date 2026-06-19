"""End-to-end coverage for the two experiment patterns, against the *shipped* demos.

The rest of the suite drives inline ``Experiment(...)`` objects; here we exercise
the real files under ``docs/`` the way a user (or the CLI) does, so a demo can't
bit-rot — a broken import, a drifted ``main(ctx)`` signature, or a ``load_experiment``
contract change fails CI instead of silently rotting the onboarding examples.

- **Memoized / detached pattern:** ``load_experiment(<file>)`` then drive the real
  ``main(ctx)`` DAG to completion on a ``LocalApparatus`` (detached subprocess
  workers + the durable memo store) — the same path ``mini run`` takes.
- **Interactive pattern:** drive an ``Apparatus`` directly (as a notebook does),
  fanning a sweep out with ``.map`` and reducing — no memo store, no CLI.

The light demos (``pipeline``, ``role-demo``) run to completion; the GPU/Modal-heavy
``gpt-sweep`` is only *loaded* (import + construct), which still catches rot cheaply.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mini.experiment import Experiment, load_experiment
from mini.local_apparatus import LocalApparatus
from mini.orchestration import tick

REPO = Path(__file__).resolve().parents[2]
DEMOS = sorted(REPO.glob('docs/*/experiment.py'))


def _drive(exp: Experiment, app: LocalApparatus, timeout: float = 60.0):
    """Tick the DAG to completion (launch detached work, resume on memo hits)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, payload = tick(exp, app)
        if done:
            return payload
        time.sleep(0.1)
    raise AssertionError(f'{exp.name} did not complete within {timeout}s')


@pytest.mark.parametrize('path', DEMOS, ids=lambda p: p.parent.name)
def test_every_demo_experiment_loads(path: Path):
    """Each docs/*/experiment.py defines a loadable, named, runnable experiment.

    Cheap guard for the heavy demos: catches import errors and the module-level
    ``experiment = Experiment(...)`` contract without running any training."""
    exp = load_experiment(path)
    assert exp.name == path.parent.name  # the dir is the experiment name by convention
    assert callable(exp.orchestration())  # main(ctx), or a sweep lowered to one map


def test_pipeline_demo_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The canonical memoized demo: load the real file and drive its prep→sweep
    DAG to completion on detached local workers, as ``mini run`` would."""
    monkeypatch.chdir(tmp_path)  # DATA_ROOT='.mini' + the volume resolve under here

    exp = load_experiment(REPO / 'docs/pipeline/experiment.py')
    payload = _drive(exp, LocalApparatus('pipeline', max_workers=3))

    assert set(payload) == {'meta', 'best', 'results'}
    assert len(payload['results']) == 3
    assert payload['best']['lr'] == 1e-2  # the toy loss bowl's minimum
    assert payload['meta']['vocab_size'] == payload['results'][0].get('vocab', payload['meta']['vocab_size'])


def test_role_demo_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The role-routing demo end to end: ``role=`` labels resolve through the
    real ``Experiment.roles`` table. Locally both roles are no-ops (local ``.w``
    ignores GPU knobs), so each step lands on 'cpu' — the point is that it *runs*."""
    monkeypatch.chdir(tmp_path)

    exp = load_experiment(REPO / 'docs/role-demo/experiment.py')
    out = _drive(exp, LocalApparatus('role-demo'))

    assert set(out) == {'probe', 'gpu'}  # both keys produced via distinct role-labelled steps


def test_interactive_apparatus_sweep_pattern():
    """The interactive pattern: drive an ``Apparatus`` directly (notebook-style) —
    fan a sweep out with ``.map`` and reduce to the best config. No memo store/CLI."""
    app = LocalApparatus('interactive', max_workers=3)

    def train(lr: float) -> dict:
        return {'lr': lr, 'loss': (lr - 1e-2) ** 2}  # bowl with its minimum at lr=1e-2

    results = list(app.map(train, [1e-3, 1e-2, 1e-1]))
    best = min(results, key=lambda r: r['loss'])

    assert len(results) == 3
    assert best['lr'] == 1e-2

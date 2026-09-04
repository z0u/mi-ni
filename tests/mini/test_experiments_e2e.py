"""End-to-end coverage for the memoized/detached experiment pattern, against the *shipped* demos.

The rest of the suite drives inline ``Experiment(...)`` objects; here we exercise the real files under ``docs/`` the way a user (or the CLI) does, so a demo can't bit-rot — a broken import, a drifted ``main(ctx)`` signature, or a ``load_experiment`` contract change fails CI instead of silently rotting the onboarding examples.

Each demo is loaded with ``load_experiment(<file>)`` and its real ``main(ctx)`` DAG driven to completion on a ``LocalApparatus`` (detached subprocess workers + the durable memo store) — the same path ``mini run`` takes.

The light demos (``pipeline``, the ``acts``/``probe`` pair) run to completion; the GPU/Modal-heavy ``ngpt-scaling`` is only *loaded* (import + construct), which still catches rot cheaply.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mini.experiment import Experiment, load_experiment
from mini.local_apparatus import LocalApparatus
from mini.orchestration import tick

REPO = Path(__file__).resolve().parents[2]
DEMOS = sorted(REPO.glob("docs/**/experiment.py"))


def _design_only(path: Path) -> bool:
    """A preregistered design: constants its report imports, with the DAG still to come.

    Marked in the module itself (`DESIGN_ONLY = True`) so the opt-out lives beside the file it describes, and disappears when the experiment is implemented.
    """
    return "DESIGN_ONLY = True" in path.read_text()


@pytest.fixture(autouse=True)
def _local_store_only(monkeypatch: pytest.MonkeyPatch):
    """Exercise the *local* project store, hermetically.

    These demos test orchestration + the on-disk store; an ambient ``MINI_STORE_BUCKET`` or ``MINI_PUBLISH_REPO`` (a configured HF bucket or publish repo) would divert their put/get to the network and break the local-CAS assertions. The bucket backend has its own coverage in ``test_hf_store``.
    """
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    monkeypatch.setenv("MINI_NO_PROJECT_CONFIG", "1")  # and the pyproject / mini.local.toml fallback, workers included


def _drive(exp: Experiment, app: LocalApparatus, timeout: float = 60.0):
    """Tick the DAG to completion (launch detached work, resume on memo hits)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, payload = tick(exp, app)
        if done:
            return payload
        time.sleep(0.1)
    raise AssertionError(f"{exp.name} did not complete within {timeout}s")


@pytest.mark.parametrize("path", DEMOS, ids=lambda p: p.parent.name)
def test_every_demo_experiment_loads(path: Path):
    """Each docs/*/experiment.py defines a loadable, named, runnable experiment.

    Cheap guard for the heavy demos: catches import errors and the module-level ``experiment = Experiment(...)`` contract without running any training."""
    if _design_only(path):
        pytest.skip("preregistered design; no DAG yet")
    exp = load_experiment(path)
    assert exp.name == path.parent.name  # the dir is the experiment name by convention
    assert callable(exp.main)  # main(ctx), or a sweep lowered to one map


def test_pipeline_demo_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The canonical memoized demo: load the real file and drive its prep→sweep DAG to completion on detached local workers, as ``mini run`` would."""
    monkeypatch.chdir(tmp_path)  # DATA_ROOT='.mini' + the volume resolve under here

    exp = load_experiment(REPO / "docs/pipeline/experiment.py")
    payload = _drive(exp, LocalApparatus("pipeline", max_workers=3))

    assert set(payload) == {"meta", "best", "results"}
    assert len(payload["results"]) == 3
    assert payload["best"]["lr"] == 1e-2  # the toy loss bowl's minimum
    # Prep really ran, and the sweep was keyed on what it produced: a pangram
    # repeated 200× is 26 letters + a space, 44 characters a repeat.
    assert payload["meta"] == {"vocab_size": 27, "n_chars": 8800}


def test_cross_experiment_artifact_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``acts`` publishes an activation cache; ``probe`` (a *separate* experiment) resolves the same bytes from the project-scoped store — no recompute, no shared volume. The acceptance test for #13/#22's artifact-sharing half."""
    monkeypatch.chdir(tmp_path)  # both experiments anchor .mini (and .mini/store) here

    acts = _drive(load_experiment(REPO / "docs/acts/experiment.py"), LocalApparatus("acts"))
    probe = _drive(load_experiment(REPO / "docs/probe/experiment.py"), LocalApparatus("probe"))

    # B read A's exact bytes: its recorded source sha is A's artifact sha.
    assert probe["source_sha"] == acts["artifact"].sha256
    assert probe["n_layers"] == acts["shards"]

    # The two experiments kept separate volumes but one shared store: B never ran
    # A's extraction (its only task is the probe), and the blobs live under .mini/store.
    probe_fns = {r.get("fn") for r in LocalApparatus("probe").memo_store().records()}
    assert probe_fns == {"probe_activations"}  # no extract_activations recompute
    assert (tmp_path / ".mini" / "store" / "cas").is_dir()

    # The handle resolves from a fresh client (a report would do exactly this).
    store = LocalApparatus("probe").store()
    cache = store.get(acts["artifact"], tmp_path / "check")
    assert len(list(cache.glob("*.npy"))) == acts["shards"]

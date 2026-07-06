"""
Probe a shared activation cache produced by another experiment.

This is the *consumer* half of the artifact-sharing demo. It does **not** extract
activations itself — it resolves the cache that ``docs/acts`` published by name
from the project-scoped store, then computes a small interpretability summary
(per-neuron mean activation, per layer). The reuse is the whole point: B reads
A's bytes straight from the content-addressed store — no recompute, no shared
volume, across the experiment boundary.

    bin/mini run docs/acts/experiment.py  --watch    # produce the cache first
    bin/mini run docs/probe/experiment.py --watch    # then probe it

The companion ``report.py`` reads this experiment's durable result, renders the
summary as a figure, and *publishes* the figure to a shareable URL — the report
is where assets go out to the web, distinct from the durable store the step writes.
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir
from mini.store import get, get_ref, put

DATASET = "tiny-shakespeare"


def probe_activations(dataset: str) -> dict:
    """Resolve the shared cache by name, summarize it, and store the summary."""
    import json

    import numpy as np

    art = get_ref(f"activations/{dataset}")
    if art is None:
        raise FileNotFoundError(f"no activation cache published for {dataset!r} — run docs/acts/experiment.py first")

    # Pull the shared tree into this step's volume (a warm checkout); the bytes
    # come from the project store, not from a recomputed prep.
    local = get(art, get_data_dir() / "acts-in")
    per_layer_means = {
        shard.stem: np.load(shard).mean(axis=0).round(4).tolist()  # per-neuron mean
        for shard in sorted(local.glob("*.npy"))
    }

    summary = {"dataset": dataset, "source_sha": art.sha256, "per_layer_means": per_layer_means}
    asset = put(json.dumps(summary).encode(), name=f"{dataset}-neuron-means.json")
    return {
        "dataset": dataset,
        "source_sha": art.sha256,  # proof we read A's exact bytes
        "n_layers": len(per_layer_means),
        "summary": asset,
    }


def main(ctx: Ctx) -> dict:
    return ctx.run(probe_activations, DATASET)


experiment = Experiment(name="probe", main=main)

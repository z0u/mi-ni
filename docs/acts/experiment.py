"""
Extract a stand-in activation cache and share it project-wide by name.

This is the *producer* half of a cross-experiment artifact-sharing demo (the
*consumer* is ``docs/probe``). A real interpretability run would cache a model's
per-layer activations — a few big files, or many small per-shard ones. Here we
synthesize a tiny deterministic stand-in so the *storage* path runs end to end
without a GPU or a large download.

The single step writes the cache to its volume, then ``put``s it into the
**project-scoped** content-addressed store as a *tree* artifact (one blob per
shard, deduped by content) and publishes a named ref. Because the store is keyed
by content and scoped to the project — not to this experiment — a *different*
experiment can resolve the exact same bytes by that name, with no recompute and
no shared volume:

    bin/mini run docs/acts/experiment.py --watch
    bin/mini run docs/probe/experiment.py --watch   # reuses what this produced

The step returns the :class:`~mini.store.Artifact` *handle* (a sha + size + name,
no location), so it pickles durably into the memo result and a downstream step's
memo key fingerprints it by content rather than by a path that may evaporate.
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir
from mini.store import put, set_ref

DATASET = "tiny-shakespeare"
N_LAYERS, N_NEURONS, N_TOKENS = 4, 16, 256


def extract_activations(dataset: str) -> dict:
    """Synthesize a per-layer activation cache, store it, and share it by name."""
    import hashlib

    import numpy as np

    # A *stable* seed (Python's str hash is per-process randomized) so the same
    # dataset yields byte-identical shards — the precondition for the content
    # hashes to coincide across runs, processes, and experiments.
    seed = int.from_bytes(hashlib.sha256(dataset.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    cache = get_data_dir() / "activations"
    cache.mkdir(parents=True, exist_ok=True)
    for layer in range(N_LAYERS):
        acts = rng.standard_normal((N_TOKENS, N_NEURONS)).astype("float32")
        np.save(cache / f"layer_{layer:02d}.npy", acts)

    # One handle for the whole directory: each shard is its own CAS blob, so an
    # identical cache anywhere in the project coincides, and a consumer can pull
    # one shard without the set.
    art = put(cache, name=f"{dataset}-activations")
    # Name it so another experiment can find it without knowing our memo key.
    set_ref(f"activations/{dataset}", art)
    return {"dataset": dataset, "artifact": art, "shards": len(art.children), "bytes": art.size}


def main(ctx: Ctx) -> dict:
    return ctx.run(extract_activations, DATASET)


experiment = Experiment(name="acts", main=main)

# title: Probe report

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mini import LocalApparatus, RunState
from mini.lit import stop
from mini.vis import themed

r"""
# Probe report

This notebook is a **report**, not the experiment. The experiment in
[`experiment.py`](./experiment.py)
resolves an activation cache that a *different* experiment
([`docs/acts`](../acts/experiment.py))
published to the project-scoped artifact store, summarizes it, and stores the
summary as a durable artifact. Run both from the command line:

```bash
bin/mini run docs/acts/experiment.py  --watch
bin/mini run docs/probe/experiment.py --watch
```

Here we read the durable summary back through its handle and render it. The
figure is **externalized** to the artifact store (not inlined), so the published
report stays light and the bytes are served from the bucket. The source links
above are written *relative*; `build_site` resolves them to their GitHub source
(a sibling report would resolve to its rendered page), so the only literal
relative URLs left in the published report are its assets.
"""

# The report reads results by experiment *name*, the same key the CLI uses.
NAME = "probe"

# Read-only: the apparatus gives us both the memo store (the result record)
# and the artifact store (the bytes the result points at). Neither ticks the
# DAG, so the report can't relaunch work.
app_ = LocalApparatus(NAME)
store, artifacts = app_.memo_store(), app_.store()
done = [r for r in store.records() if r.get("fn") == "probe_activations" and r.get("state") == RunState.DONE]
result = store.result(done[0]["key"]) if done else None

if result is None:
    stop(
        "Nothing to report yet. Produce the cache, then probe it:\n\n"
        "```bash\nbin/mini run docs/acts/experiment.py --watch\n"
        "bin/mini run docs/probe/experiment.py --watch\n```"
    )

rf"""
Probed **{result["dataset"]}** across **{result["n_layers"]}** layers, reading
activation bytes `{result["source_sha"][:12]}…` straight from the shared
store — no recompute.
"""

# Resolve the durable summary artifact (its handle rode along in the result).
dest = Path(tempfile.mkdtemp()) / result["summary"].name
summary = json.loads(artifacts.get(result["summary"], dest).read_text())
means = {layer: np.array(vals) for layer, vals in summary["per_layer_means"].items()}


# `use_publisher` (installed by the lit runner) routes this figure out to a file
# beside the exported HTML; the rendered <img> points at its relative URL instead
# of carrying the PNG inline, so the exported report stays light.
@themed(
    name="mean-activation",  # externalized assets save as mean-activation-{light,dark}.png
    alt_text=(
        "A heatmap of mean activation per neuron (x-axis) for each layer (y-axis) of the "
        "stand-in activation cache. Values cluster near zero, as expected for a synthesized "
        "standard-normal stand-in, with small layer-to-layer variation."
    ),
)
def plot_mean_activation() -> plt.Figure:
    grid = np.vstack([means[layer] for layer in sorted(means)])
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu", vmin=-0.3, vmax=0.3)
    ax.set(xlabel="neuron", ylabel="layer", yticks=range(len(means)), title="Mean activation per neuron")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # No tight_layout(): the base style enables constrained_layout, which
    # handles the colorbar; calling both clashes the layout engines.
    return fig


plot_mean_activation()

# title: Pipeline report

import matplotlib.pyplot as plt

from mini import LocalApparatus, RunState
from mini.lit import stop
from mini.vis import themed

r"""
# Pipeline report

This page is a **report**, not the experiment. The experiment is defined
in [`experiment.py`](./experiment.py) as an importable `main(ctx)` DAG, and
run from the command line:

```bash
bin/mini run docs/pipeline/experiment.py --watch --workers 3
```

That writes durable, content-addressed results to a memo store. This page
reads them back and renders them — it never launches or re-runs the work, so
it opens standalone (no GPU, no waiting) and shows the last run's results.
"""

# The report reads results by experiment *name*, the same key the CLI uses.
NAME = "pipeline"

# Read-only: pull the per-task records straight off the durable store. Getting
# the store from the apparatus (rather than ticking the DAG) is what keeps this
# a *report* — it can't accidentally relaunch work.
store = LocalApparatus(NAME).memo_store()
records = store.records()
runs = sorted(
    (store.result(r["key"]) for r in records if r.get("fn") == "train" and r.get("state") == RunState.DONE),
    key=lambda d: d["lr"],
)

if not records:
    stop(
        "Nothing to report yet. Run the experiment first:\n\n"
        "```bash\nbin/mini run docs/pipeline/experiment.py --watch --workers 3\n```"
    )

best = min(runs, key=lambda d: d["val_loss"]) if runs else None

rf"""
{
    f"**Best config:** `lr={best['lr']:g}` → val_loss **{best['val_loss']}** (swept {len(runs)} learning rates)."
    if best
    else "_The sweep has not finished — check `bin/mini status pipeline`._"
}
"""

glyph = {RunState.DONE: "✓", RunState.RUNNING: "▸", RunState.FAILED: "✗", RunState.CANCELLED: "⊘"}
header = "| task | key | state | metrics |\n| --- | --- | --- | --- |"
rows = [
    f"| {r.get('fn', 'task')} | `{r['key']}` | {glyph.get(r.get('state'), '·')} {r.get('state', 'pending')} "
    f"| {'  '.join(f'{k}={v:g}' for k, v in (r.get('metrics') or {}).items())} |"
    for r in records
]

rf"""
{"\n".join([header, *rows])}
"""

if not runs:
    stop()


@themed(
    alt_text=(
        "Final validation loss against learning rate on a log x-axis. The curve is U-shaped: "
        "the middle learning rate reaches the lowest loss, and the extremes do worse — so the "
        "sweep has a clear best in the middle."
    )
)
def plot_sweep() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    lrs = [d["lr"] for d in runs]
    losses = [d["val_loss"] for d in runs]
    ax.plot(lrs, losses, "o-", color="tab:blue")
    best = min(runs, key=lambda d: d["val_loss"])
    ax.plot(best["lr"], best["val_loss"], "o", color="tab:red", markersize=11, fillstyle="none", label="best")
    ax.set(xscale="log", xlabel="learning rate", ylabel="validation loss", title="Learning-rate sweep")
    ax.legend()
    fig.tight_layout()
    return fig


plot_sweep()

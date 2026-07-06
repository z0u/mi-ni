import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup(hide_code=True):
    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt

    from mini import LocalApparatus, RunState
    from mini.reports import report_bundle, use_publisher
    from mini.vis import themed

    # The report reads results by experiment *name*, the same key the CLI uses.
    NAME = "pipeline"

    # Externalize every themed figure to a file beside the exported HTML, referenced
    # by a relative URL — keeps the report light, and `build_site` repoints those URLs
    # at the bucket (one <base> tag) when publishing. No publisher → figures inline.
    use_publisher(report_bundle(__file__))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Pipeline report

    This notebook is a **report**, not the experiment. The experiment is defined
    in [`experiment.py`](./experiment.py) as an importable `main(ctx)` DAG, and
    run from the command line:

    ```bash
    bin/mini run docs/pipeline/experiment.py --watch --workers 3
    ```

    That writes durable, content-addressed results to a memo store. This notebook
    reads them back and renders them — it never launches or re-runs the work, so
    it opens standalone (no GPU, no waiting) and shows the last run's results.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Read-only: pull the per-task records straight off the durable store. Getting
    # the store from the apparatus (rather than ticking the DAG) is what keeps this
    # a *report* — it can't accidentally relaunch work.
    store = LocalApparatus(NAME).memo_store()
    records = store.records()
    runs = sorted(
        (store.result(r["key"]) for r in records if r.get("fn") == "train" and r.get("state") == RunState.DONE),
        key=lambda d: d["lr"],
    )
    return records, runs


@app.cell(hide_code=True)
def _(records, runs):
    mo.stop(
        not records,
        mo.md(
            "Nothing to report yet. Run the experiment first:\n\n"
            "```bash\nbin/mini run docs/pipeline/experiment.py --watch --workers 3\n```"
        ),
    )
    best = min(runs, key=lambda d: d["val_loss"]) if runs else None
    mo.md(
        f"**Best config:** `lr={best['lr']:g}` → val_loss **{best['val_loss']}** (swept {len(runs)} learning rates)."
        if best
        else "_The sweep has not finished — check `bin/mini status pipeline`._"
    )
    return


@app.cell(hide_code=True)
def _(records):
    mo.stop(not records)
    glyph = {RunState.DONE: "✓", RunState.RUNNING: "▸", RunState.FAILED: "✗", RunState.CANCELLED: "⊘"}
    header = "| task | key | state | metrics |\n| --- | --- | --- | --- |"
    rows = [
        f"| {r.get('fn', 'task')} | `{r['key']}` | {glyph.get(r.get('state'), '·')} {r.get('state', 'pending')} "
        f"| {'  '.join(f'{k}={v:g}' for k, v in (r.get('metrics') or {}).items())} |"
        for r in records
    ]
    mo.md("\n".join([header, *rows]))
    return


@app.cell(hide_code=True)
def _(runs):
    mo.stop(not runs)

    @themed(
        alt_text=(
            "Final validation loss against learning rate on a log x-axis. The curve is U-shaped: "
            "the middle learning rate reaches the lowest loss, and the extremes do worse — so the "
            "sweep has a clear best in the middle."
        )
    )
    def _plot() -> plt.Figure:
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

    mo.Html(_plot())
    return


if __name__ == "__main__":
    app.run()

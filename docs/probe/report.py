import marimo

__generated_with = '0.23.3'
app = marimo.App(width='medium', auto_download=['html'])

with app.setup(hide_code=True):
    import json

    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    from mini import LocalApparatus, RunState
    from mini.vis import themed

    # The report reads results by experiment *name*, the same key the CLI uses.
    NAME = 'probe'


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Probe report

    This notebook is a **report**, not the experiment. The experiment in
    [`experiment.py`](./experiment.py) resolves an activation cache that a
    *different* experiment ([`docs/acts`](../acts/experiment.py)) published to the
    project-scoped artifact store, summarizes it, and stores the summary as a
    durable artifact. Run both from the command line:

    ```bash
    bin/mini run docs/acts/experiment.py  --watch
    bin/mini run docs/probe/experiment.py --watch
    ```

    Here we read the durable summary back through its handle, render it, and
    **publish** the figure to a shareable URL — the report is where assets go out
    to the web, distinct from the content-addressed store the step writes to.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Read-only: the apparatus gives us both the memo store (the result record)
    # and the artifact store (the bytes the result points at). Neither ticks the
    # DAG, so the report can't relaunch work.
    app_ = LocalApparatus(NAME)
    store, artifacts = app_.memo_store(), app_.store()
    done = [r for r in store.records() if r.get('fn') == 'probe_activations' and r.get('state') == RunState.DONE]
    result = store.result(done[0]['key']) if done else None
    return artifacts, result, store


@app.cell(hide_code=True)
def _(result, store):
    mo.stop(
        result is None,
        mo.md(
            'Nothing to report yet. Produce the cache, then probe it:\n\n'
            '```bash\nbin/mini run docs/acts/experiment.py --watch\n'
            'bin/mini run docs/probe/experiment.py --watch\n```'
        ),
    )
    mo.md(
        f'Probed **{result["dataset"]}** across **{result["n_layers"]}** layers, '
        f'reading activation bytes `{result["source_sha"][:12]}…` straight from the '
        'shared store — no recompute.'
    )
    return


@app.cell(hide_code=True)
def _(artifacts, result):
    mo.stop(result is None)
    # Resolve the durable summary artifact (its handle rode along in the result).
    import tempfile
    from pathlib import Path

    dest = Path(tempfile.mkdtemp()) / result['summary'].name
    summary = json.loads(artifacts.get(result['summary'], dest).read_text())
    means = {layer: np.array(vals) for layer, vals in summary['per_layer_means'].items()}
    return (means,)


@app.cell(hide_code=True)
def _(artifacts, means, result):
    mo.stop(result is None)

    @themed(
        alt_text=(
            'A heatmap of mean activation per neuron (x-axis) for each layer (y-axis) of the '
            'stand-in activation cache. Values cluster near zero, as expected for a synthesized '
            'standard-normal stand-in, with small layer-to-layer variation.'
        )
    )
    def _plot() -> plt.Figure:
        grid = np.vstack([means[layer] for layer in sorted(means)])
        fig, ax = plt.subplots(figsize=(6, 3))
        im = ax.imshow(grid, aspect='auto', cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax.set(xlabel='neuron', ylabel='layer', yticks=range(len(means)), title='Mean activation per neuron')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # No tight_layout(): the base style enables constrained_layout, which
        # handles the colorbar; calling both clashes the layout engines.
        return fig

    html = _plot()

    # Publish a standalone PNG of the figure to a shareable URL. Locally this is a
    # file:// URL under the project store; a web-reachable backend (e.g. an HF
    # bucket) returns an https:// resolve URL for the same handle — see todo.md.
    from io import BytesIO

    grid = np.vstack([means[layer] for layer in sorted(means)])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(grid, aspect='auto', cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax.set_axis_off()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    asset = artifacts.put(buf.getvalue(), name=f'{result["dataset"]}-neuron-means.png')
    url = artifacts.publish(asset, f'reports/probe/{result["dataset"]}-neuron-means.png')

    mo.md(f'{html}\n\nPublished figure: [`{url}`]({url})')
    return


if __name__ == '__main__':
    app.run()

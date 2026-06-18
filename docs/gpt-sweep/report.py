import marimo

__generated_with = '0.23.3'
app = marimo.App(width='medium', auto_download=['html'])

with app.setup(hide_code=True):
    import json
    from pathlib import Path

    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt

    from mini.vis import themed

    # Sweep axes (kept in sync with experiment.py), and per-arch plot colours.
    LRS = ['3e-3', '1e-2', '4e-2']
    ARCHS = ['baseline', 'nGPT', 'nGPT (scalar)']
    ARCH_COLORS = {'baseline': 'tab:gray', 'nGPT': 'tab:red', 'nGPT (scalar)': 'tab:green'}

    try:
        _here = Path(__file__).resolve().parent
    except NameError:
        _here = Path('docs/gpt-sweep')
    RESULTS_PATH = _here / 'results.json'


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Architecture sweep: GPT versus nGPT

    A controlled comparison of the baseline LayerNorm GPT against
    [nGPT](https://arxiv.org/abs/2410.01131), swept across three peak learning
    rates — nine training runs in all.

    This is a **report**: it reads results the experiment already produced and
    renders them. The experiment itself is [`experiment.py`](./experiment.py), an
    importable `main(ctx)` DAG (one data-prep step, then a nine-cell GPU sweep)
    run from the CLI on Modal L4s:

    ```bash
    bin/mini run docs/gpt-sweep/experiment.py --app modal --gpu L4 --max-containers 9 --timeout 600
    ```

    The val-loss curves below were gathered from that run into
    [`results.json`](./results.json), so this notebook opens standalone — no GPU,
    no Modal, no waiting.
    """)
    return


@app.cell(hide_code=True)
def _():
    curves = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    return (curves,)


@app.cell(hide_code=True)
def _(curves):
    mo.stop(
        not curves,
        mo.md(
            'No results yet. Run the experiment first:\n\n'
            '```bash\nbin/mini run docs/gpt-sweep/experiment.py --app modal --gpu L4 --max-containers 9 --timeout 600\n```'
            '\n\nThen snapshot the gathered curves to `results.json` (see the experiment README).'
        ),
    )
    flat = {(a, lr): v for a in ARCHS for lr in LRS if (v := curves.get(f'{a}|{lr}'))}
    best_arch, best_lr = min(flat, key=lambda k: min(flat[k]))
    mo.md(
        f'**Best run:** {best_arch} at peak LR {best_lr} — val_loss '
        f'**{min(flat[(best_arch, best_lr)]):.2f}** over {len(flat)} completed cells.'
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Does normalization help? nGPT versus the LayerNorm baseline

    nGPT added a fair amount of machinery, so it's worth checking how it compares
    against the baseline. The two architectures don't want the same learning rate:
    normalization rescales the effective gradients, which shifts nGPT's useful LR
    band upward. To isolate the architecture, we sweep both across the same three
    peak learning rates, with the same warmup-then-cosine schedule, on the same
    data. The only difference within each panel is LayerNorm versus the hypersphere.

    - **baseline** — standard pre-norm transformer (LayerNorm + additive residual).
    - **nGPT** — normalized transformer as published (per-channel eigen learning rates).
    - **nGPT (scalar)** — a single learnable scalar gate per sub-module instead of
      per-channel weights.
    """)
    return


@app.cell(hide_code=True)
def _(curves):
    mo.stop(not curves)

    @themed(
        alt_text='Three panels, one per peak learning rate. Each plots validation loss versus epoch '
        'for the baseline, nGPT, and nGPT (scalar) architectures, with the minimum of each curve marked. '
        'At the lowest learning rate the baseline edges ahead; at higher rates the two nGPT variants reach '
        'slightly lower loss and track each other almost exactly.'
    )
    def plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for ax, lr in zip(axes, LRS, strict=True):
            for arch in ARCHS:
                ys = curves.get(f'{arch}|{lr}')
                if not ys:
                    continue
                ax.plot(range(1, len(ys) + 1), ys, color=ARCH_COLORS[arch], lw=1.5, label=arch)
                best = min(range(len(ys)), key=ys.__getitem__)
                ax.scatter([best + 1], [ys[best]], color=ARCH_COLORS[arch], s=18, zorder=5)
            ax.set_title(f'peak LR = {lr}')
            ax.set_xlabel('epoch')
            ax.grid(alpha=0.3)
        axes[0].set_ylabel('val_loss')
        axes[0].legend()
        fig.tight_layout()
        return fig

    mo.Html(plot())
    return


@app.cell(hide_code=True)
def _(curves):
    mo.stop(not curves)

    def _cell(arch: str, lr: str) -> str:
        ys = curves.get(f'{arch}|{lr}')
        return f'{min(ys):.2f}' if ys else '—'

    # Bold the best (lowest) architecture in each LR column.
    def _row(lr: str) -> str:
        vals = {a: min(curves[f'{a}|{lr}']) for a in ARCHS if curves.get(f'{a}|{lr}')}
        best = min(vals, key=vals.get) if vals else None
        cells = ' | '.join(f'**{_cell(a, lr)}**' if a == best else _cell(a, lr) for a in ARCHS)
        return f'| {lr} | {cells} |'

    table = '\n'.join(
        ['| peak LR | baseline | nGPT | nGPT (scalar) |', '| --- | --- | --- | --- |', *(_row(lr) for lr in LRS)]
    )
    mo.md(f"""
    Best (minimum) validation loss per cell:

    {table}

    Two things stand out. **nGPT needs a higher learning rate to pay off** — at the
    lowest rate it trails the baseline, but given enough LR it reaches a slightly
    lower loss. And the **scalar-gate simplification matches the full per-channel
    variant** almost exactly: the per-channel granularity buys nothing at this scale.

    The paper's headline "trains several times faster" and its stability advantages
    don't reproduce at this toy scale — the LayerNorm baseline never threatens to
    diverge — so at this scale nGPT is a small but real improvement in final loss.
    The load-bearing piece is the scalar gate on the normalized residual (see the
    [model notes](../../src/experiment/model/README.md)); the rest of nGPT's
    machinery is optional here.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## References

    Loshchilov, I., Hsieh, C.-P., Sun, S., & Ginsburg, B. (2024). nGPT: Normalized
    transformer with representation learning on the hypersphere. _arXiv_.
    https://arxiv.org/abs/2410.01131

    Karpathy, A. (2022). nanoGPT [Computer software]. GitHub.
    https://github.com/karpathy/nanoGPT
    """)
    return


if __name__ == '__main__':
    app.run()

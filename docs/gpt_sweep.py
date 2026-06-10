import marimo

__generated_with = '0.23.3'
app = marimo.App(width='medium', auto_download=['html'])

with app.setup(hide_code=True):
    import logging

    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt

    from experiment.config import (
        DataConfig,
        MixedPrecisionConfig,
        ModelConfig,
        OptimizerConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )
    from experiment.utils import align
    from mini import LocalApparatus, ModalApparatus, get_data_dir  # noqa: F401
    from mini.logging import SimpleLoggingConfig
    from mini.vis import themed
    from utils.time import duration as t

    logging_config = SimpleLoggingConfig().info('notebook', 'experiment', 'mini', 'utils')
    logging_config.apply()

    log = logging.getLogger('notebook')


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Architecture sweep: GPT versus nGPT

    A controlled comparison of the baseline LayerNorm GPT against nGPT, swept
    across three peak learning rates. Each cell in the grid runs as an
    independent training job; on Modal all cells launch concurrently in separate
    containers.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configuration
    """)
    return


@app.cell(hide_code=True)
def _():
    # Axes of the sweep.
    LRS = [('3e-3', 3e-3), ('1e-2', 1e-2), ('4e-2', 4e-2)]
    ARCH_CFGS = [
        ('baseline', dict(architecture='gpt')),
        ('nGPT', dict(architecture='ngpt', ngpt_variant='full')),
        ('nGPT (scalar)', dict(architecture='ngpt', ngpt_variant='crude')),
    ]
    ARCH_COLORS = {
        'baseline': 'tab:gray',
        'nGPT': 'tab:red',
        'nGPT (scalar)': 'tab:green',
    }
    return ARCH_CFGS, ARCH_COLORS, LRS


@app.cell(hide_code=True)
def sweep_config(ARCH_CFGS, LRS, is_headless, run_button):
    def _make_config(lr_float, arch_kwargs):
        is_ngpt = arch_kwargs.get('architecture') == 'ngpt'
        return TrainingConfig(
            model=ModelConfig(
                vocab_size=64,  # updated after data prep
                block_size=512,
                n_embd=32,
                n_head=8,
                n_head_dim=8,
                n_ff=128,
                n_layer=12,
                dropout=0 if is_ngpt else 0.1,
                **arch_kwargs,
            ),
            tokenizer=TokenizerConfig(vocabulary=[]),
            data=DataConfig(
                batch_size=16,
                oversample=1,
                train_split=0.8,
                padding_chance=0.1,
            ),
            optimizer=OptimizerConfig(
                weight_decay=0 if is_ngpt else 1e-3,
                learning_rate=lr_float,
                betas=(0.9, 0.95),
            ),
            scheduler=SchedulerConfig(
                epochs=100,
                warmup_epochs=10,
                min_lr_factor=0.01,
            ),
            amp=MixedPrecisionConfig(enabled=False),
        )

    mo.stop(not run_button.value and not is_headless)

    # One (config, arch_label, lr_str) triple per sweep cell.
    sweep = [
        (_make_config(lr_float, arch_kwargs), arch_label, lr_str)
        for lr_str, lr_float in LRS
        for arch_label, arch_kwargs in ARCH_CFGS
    ]
    return (sweep,)


@app.cell(hide_code=True)
def _(app_type, run_button):
    mo.md(f"""
    {app_type} {run_button}
    """)
    return


@app.cell(hide_code=True)
def _(app_type, sweep):
    if app_type.value == 'local':
        sweep_app = LocalApparatus('arch-sweep')
    elif app_type.value == 'modal':
        sweep_app = (
            ModalApparatus('arch-sweep')
            .w(
                gpu='L4',
                max_containers=len(sweep),
                timeout=int(t('15 min')),
            )
            .before_each(logging_config.apply)
        )
    else:
        raise ValueError(f'Unknown apparatus {app_type.value}')

    mo.md(f'Using **{sweep_app}**')
    return (sweep_app,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data

    Same Pride and Prejudice corpus as the single-arch notebook. If data is
    already prepared in the volume (e.g. from running `nanogpt.py` first), this
    is a fast no-op.
    """)
    return


@app.function(hide_code=True)
def download_pride_and_prejudice():
    """Download Pride and Prejudice from the Gutenberg HuggingFace dataset."""
    import ftfy
    import pandas as pd

    from experiment.config import DatasetMetadata

    url = 'https://huggingface.co/api/datasets/larenwell/book-gutenberg-train/parquet/default/train/0.parquet'
    df = pd.read_parquet(url, columns=['text'])
    text = df.iloc[0]['text']
    text, explanation = ftfy.fix_and_explain(text)
    metadata = DatasetMetadata(
        title='Pride and Prejudice',
        author='Jane Austen',
        url=url,
        fixes=explanation or [],
        total_chars=len(text),
    )
    return text, metadata


@app.function(hide_code=True)
def prepare_data():
    """Download, tokenize, and save training data to the volume."""
    from experiment.compute.data_pipelines import save_data
    from experiment.data.preparation import tokenize_data

    data_dir = get_data_dir()
    sources = [download_pride_and_prejudice()]
    data, metadata = tokenize_data(sources)
    save_data(data, metadata, data_dir)
    return metadata


@app.cell(hide_code=True)
async def _(sweep, sweep_app):
    input_metadata = await sweep_app.arun(prepare_data)
    for _cfg, _, _ in sweep:
        _cfg.tokenizer = input_metadata.tokenizer_config.model_copy()
        _cfg.model.vocab_size = align(_cfg.tokenizer.vocab_size, 64)
    input_metadata.model_dump(exclude={'tokenizer_config'})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Does normalization help? nGPT versus the LayerNorm baseline

    nGPT added a fair amount of machinery, so it's worth checking how it
    compares against the baseline. The two architectures don't want the same
    learning rate: normalization rescales the effective gradients, which shifts
    nGPT's useful LR band upward. To isolate the architecture, we sweep both
    models across the same three peak learning rates, each with the same
    warmup-then-cosine schedule, on the same data. The only difference within
    each panel below is LayerNorm versus the hypersphere.

    We test three variants:
    - **baseline** — standard pre-norm transformer (LayerNorm + additive residual).
    - **nGPT** — normalized transformer as published (per-channel eigen learning rates).
    - **nGPT (scalar)** — simplified variant: a single learnable scalar gate per
      sub-module instead of per-channel weights.
    """)
    return


@app.function(hide_code=True)
def train_one(config, arch_label, lr_str):
    """Train one sweep run; return the arch label, LR string, and per-epoch val losses."""
    from experiment.compute.training import train_model

    _, metrics = train_model(config, get_data_dir())
    return arch_label, lr_str, [m.val_loss for m in metrics]


@app.cell(hide_code=True)
async def _(sweep, sweep_app):
    _results = [
        x
        async for x in sweep_app.amap(
            train_one,
            [c for c, _, _ in sweep],
            [a for _, a, _ in sweep],
            [lr for _, _, lr in sweep],
        )
    ]
    SWEEP = {(arch, lr): losses for arch, lr, losses in _results}
    return (SWEEP,)


@app.cell(hide_code=True)
def _(ARCH_CFGS, ARCH_COLORS, LRS, SWEEP):
    @themed
    def plot():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for ax, (lr_str, _) in zip(axes, LRS, strict=True):
            for arch_label, _ in ARCH_CFGS:
                ys = SWEEP.get((arch_label, lr_str), [])
                if not ys:
                    continue
                color = ARCH_COLORS[arch_label]
                ax.plot(range(1, len(ys) + 1), ys, color=color, lw=1.5, label=arch_label)
                best = min(range(len(ys)), key=ys.__getitem__)
                ax.scatter([best + 1], [ys[best]], color=color, s=18, zorder=5)
            ax.set_title(f'peak LR = {lr_str}')
            ax.set_xlabel('epoch')
            ax.grid(alpha=0.3)
        axes[0].set_ylabel('val_loss')
        axes[0].legend()
        return fig

    mo.Html(plot())
    return


@app.cell
def _(SWEEP):
    for (_label, _lr), _losses in SWEEP.items():
        print(f'{_label} @ {_lr}: {min(_losses):.2f}')
    return


@app.cell(hide_code=True)
def _(SWEEP):
    mo.md(rf"""
    It seems that nGPT needs a higher learning rate to pay off. At 3e-3 (left
    panel) it is slightly worse than the baseline. But given enough LR, nGPT
    reaches a slightly lower loss. The scalar-gate simplification (`nGPT
    (scalar)`) matches the full per-channel variant almost exactly — the
    per-channel granularity buys nothing at this scale:

    | peak LR | baseline | nGPT | nGPT (scalar) |
    | --- | --- | --- | --- |
    | 3e-3 | **{min(SWEEP[('baseline', '3e-3')]):.2f}** | {min(SWEEP[('nGPT', '3e-3')]):.2f} | {min(SWEEP[('nGPT (scalar)', '3e-3')]):.2f} |
    | 1e-2 | {min(SWEEP[('baseline', '1e-2')]):.2f} | **{min(SWEEP[('nGPT', '1e-2')]):.2f}** | {min(SWEEP[('nGPT (scalar)', '1e-2')]):.2f} |
    | 4e-2 | {min(SWEEP[('baseline', '4e-2')]):.2f} | {min(SWEEP[('nGPT', '4e-2')]):.2f} | **{min(SWEEP[('nGPT (scalar)', '4e-2')]):.2f}** |

    However, it does not train dramatically faster here. In the 1e-2 panel, nGPT
    descends *slower* than the baseline for the first ten epochs, crossing below
    only around epoch 20 before settling lower. The paper's headline "trains
    several times faster" result doesn't reproduce at this toy scale, nor does
    its stability advantage, since the LayerNorm baseline never threatens to
    diverge. Both effects are expected to appear only with larger models, longer
    context, and more aggressive learning rates.

    So at this scale nGPT is a small, real improvement in final loss — but the
    shape of the latent embedding manifold is potentially more interesting than
    the loss curve.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How much of the machinery is load-bearing?

    nGPT bundles several separable ideas, and we only really want one of them:
    the normalized residual stream. During development we tried stripping it
    to the bone — removing the scalar gate entirely and using a plain additive
    retraction $h \leftarrow \text{Norm}(h + \text{sub}(h))$. The result flatlined
    at val_loss ≈ 3.08 at every learning rate, barely below $\ln(64) \approx 4.16$
    (the uniform-distribution score). The model had learned little more than letter
    frequencies.

    The culprit is the missing gate, not the scalar simplification. With
    unit-norm weights and a unit-norm input, each sub-module's output already
    has norm $\approx 1$ — the same scale as the residual it joins. So
    $\text{Norm}(h + \text{sub}(h))$ rotates the hidden state by roughly **45°
    per layer**: after twelve layers the original token identity is completely
    overwritten, and there is nothing left for the residual stream to carry.
    nGPT's $\alpha$, initialised at 0.05, is what holds each step down to a few
    degrees. It turns out it's the load-bearing piece that makes a *deep*
    normalized stack trainable at all.

    Adding the gate back as a single learnable **scalar** $\alpha$ per
    sub-module (a ReZero/LayerScale-style gate) recovers essentially all of
    nGPT's performance, as the `nGPT (scalar)` curves above show. The
    per-channel granularity of the original eigen learning rates shows no
    benefit at this scale.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Where does the gate settle?

    Because the gate is a single number per sub-module, we can read it back
    after training (`NGPT.scale_report()`). A natural guess is that, to keep a
    twelve-layer stack from blowing past the sphere, each step should be worth
    about $1/n_\text{layer}$ — so the whole stack moves an $O(1)$ distance
    overall. The mean gate lands close to that line:

    | peak LR | mean $\alpha$ | $1/n_\text{layer}$ |
    | --- | --- | --- |
    | 3e-3 | 0.086 | 0.083 |
    | 1e-2 | 0.088 | 0.083 |
    | 4e-2 | 0.070 | 0.083 |

    And note it had to *move there*: every gate was initialised at 0.05, and
    training pulled the average up toward $1/n_\text{layer}$. The attention
    gates ($\alpha_a$) run consistently larger and more variable than the MLP
    gates ($\alpha_m$), and at the most aggressive learning rate a few layers
    collapse their gate toward zero while others spike — the clean
    $1/n_\text{layer}$ story is a property of the *mean* step, not a uniform
    setting across depth.
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


@app.cell(hide_code=True)
def options():
    app_type = mo.ui.radio(
        label='Apparatus',
        options=['local', 'modal'],
        value=mo.cli_args().get('app', 'local'),
        inline=True,
    )
    run_button = mo.ui.run_button(
        label='Run',
    )
    is_headless = mo.app_meta().request is None
    return app_type, is_headless, run_button


if __name__ == '__main__':
    app.run()

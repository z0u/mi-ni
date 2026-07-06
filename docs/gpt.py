import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup(hide_code=True):
    import logging
    from functools import partial

    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt

    from experiment.config import (
        DataConfig,
        ModelConfig,
        OptimizerConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )
    from experiment.utils import align
    from mini import LocalApparatus, ModalApparatus, get_data_dir  # noqa: F401
    from mini.logging import SimpleLoggingConfig
    from mini.reports import report_bundle, use_publisher
    from mini.vis import themed
    from utils.lr_finder.vis import plot_lr_finder
    from utils.time import duration as t

    logging_config = SimpleLoggingConfig().info("notebook", "experiment", "mini", "utils")
    logging_config.apply()

    log = logging.getLogger("notebook")

    # mini:source-only — this notebook trains inline (a full run on every execution), so
    # it doesn't fit the read-from-store report model the site build assumes. It's excluded
    # from the published report set: the build never runs it, and links to it (e.g. from
    # docs/index.md) resolve to its GitHub source rather than a rendered page. Run it
    # interactively with `./go open docs/gpt.py` (pick the Modal apparatus for the GPU).

    # Externalize every themed figure to a file beside the exported HTML, referenced
    # by a relative URL — keeps the report light, and `build_site` repoints those URLs
    # at the bucket (one <base> tag) when publishing. No publisher → figures inline.
    use_publisher(report_bundle(__file__))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Character-level GPT

    This experiment trains a tiny transformer on character-level data, based on a port
    of [nanoGPT](https://github.com/karpathy/nanoGPT). Most of the code lives in
    modules under [src/experiment](../src/experiment); this notebook ties it together.
    """)
    return


@app.cell(hide_code=True)
def _(app_type, arch, ngpt_variant, run_button):
    mo.md(f"""
    {arch} {ngpt_variant if arch.value == "ngpt" else ""}

    {app_type} {run_button}
    """)
    return


@app.cell(hide_code=True)
def configuration(arch, is_headless, ngpt_variant, run_button):
    mo.stop(not run_button.value and not is_headless)

    _is_ngpt = arch.value == "ngpt"
    config = TrainingConfig(
        model=ModelConfig(
            vocab_size=64,  # set after loading the dataset
            block_size=512,
            n_embd=32,
            n_head=8,
            n_head_dim=8,
            n_ff=128,
            n_layer=12,
            dropout=0 if _is_ngpt else 0.1,
            architecture=arch.value,
            ngpt_variant=ngpt_variant.value,
        ),
        tokenizer=TokenizerConfig(vocabulary=[]),
        data=DataConfig(
            batch_size=16,
            oversample=2,
            train_split=0.8,
            padding_chance=0.1,
        ),
        optimizer=OptimizerConfig(
            weight_decay=0 if _is_ngpt else 1e-3,
            learning_rate=0,  # set by LR finder
            betas=(0.9, 0.95),
        ),
        scheduler=SchedulerConfig(
            epochs=100,
            warmup_epochs=10,
            min_lr_factor=0.01,
        ),
    )
    return (config,)


@app.cell(hide_code=True)
def apparatus(app_type, is_headless, run_button):
    mo.stop(not run_button.value and not is_headless)

    if app_type.value == "local":
        app = LocalApparatus("nanogpt")
    elif app_type.value == "modal":
        app = (
            ModalApparatus("nanogpt")
            .w(
                gpu="L4",
                max_containers=1,
                timeout=int(t("30 min")),  # cold L4: JIT-compile + a full 100-epoch run
            )
            .before_each(logging_config.apply)
        )
    else:
        raise ValueError(f"Unknown apparatus {app_type.value}")

    mo.md(f"Using **{app}**")
    return (app,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data

    We'll grab a book from a [HuggingFace mirror of Project Gutenberg](https://huggingface.co/datasets/larenwell/book-gutenberg-train). It's just one big block of
    text from which we take random substrings. These may overlap, but we aim to take
    roughly the entire corpus on each epoch.

    Of note: the "labels" $y$ are the same as the input $x$, shifted by one, since
    we want to predict each next token.

    ```python
    x = data[s : s + block_size]
    y = data[s + 1 : s + block_size + 1]
    ```
    """)
    return


@app.function(hide_code=True)
def download_pride_and_prejudice():
    """Download Pride and Prejudice from the Gutenberg HuggingFace dataset."""
    import ftfy
    import pandas as pd

    from experiment.config import DatasetMetadata

    url = "https://huggingface.co/api/datasets/larenwell/book-gutenberg-train/parquet/default/train/0.parquet"
    df = pd.read_parquet(url, columns=["text"])
    text = df.iloc[0]["text"]
    text, explanation = ftfy.fix_and_explain(text)
    metadata = DatasetMetadata(
        title="Pride and Prejudice",
        author="Jane Austen",
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


@app.cell
async def _(app, config):
    input_metadata = await app.arun(prepare_data)

    config.tokenizer = input_metadata.tokenizer_config.model_copy()
    config.model.vocab_size = align(config.tokenizer.vocab_size, 64)

    input_metadata.model_dump(exclude={"tokenizer_config"})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Learning rate search

    Before training, we run a multi-scale learning rate range test. The finder
    progressively narrows the search space to improve stability.
    """)
    return


@app.function(hide_code=True)
def find_learning_rate(config):
    """Run a multi-scale LR range test and return (lr, config, history)."""
    import jax.random as jr
    import numpy as np

    from experiment.compute.data_pipelines import load_data
    from experiment.data.batches import sample_batches, split_data
    from experiment.model.gpt import GPT
    from experiment.training.loop import loss_fn
    from experiment.training.optimizer import configure_optimizer
    from utils.lr_finder.lr_finder import lr_finder_search

    data_dir = get_data_dir()
    model = GPT(config.model, key=jr.key(config.seed))
    data, _ = load_data(data_dir)
    train_data, _ = split_data(data, config.data.train_split)
    rng = np.random.default_rng(config.seed)

    def batches():
        while True:
            yield from sample_batches(train_data, config.data, config.model, 100, rng)

    return lr_finder_search(
        model,
        loss_fn,
        lambda learning_rate: configure_optimizer(model, config.optimizer, learning_rate),
        batches(),
        key=jr.key(config.seed),
    )


@app.cell(hide_code=True)
async def _(app, config):
    suggested_lr, lr_config, lr_history = await app.arun(find_learning_rate, config)

    config.optimizer.learning_rate = suggested_lr
    mo.output.append(mo.md(f"Suggested learning rate: **{suggested_lr:.2e}**"))
    return lr_config, lr_history


@app.cell(hide_code=True)
def _(lr_config, lr_history):
    mo.Html(
        themed(plot_lr_finder, alt_text="Learning-rate finder plot")(
            lr_history,
            lr_config,
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training

    Now that we have a good learning rate, let's do a full training run. Checkpoints
    are saved to the volume periodically.
    """)
    return


@app.function(hide_code=True)
def train(config):
    """Run a full training loop. Return per-epoch metrics."""
    from experiment.compute.training import train_model

    data_dir = get_data_dir()
    _, metrics = train_model(config, data_dir)
    return metrics


@app.cell
async def _(app, config):
    training_metrics = await app.arun(train, config)
    return (training_metrics,)


@app.cell(hide_code=True)
def _(training_metrics):
    # Plot training curve
    epochs = [m.epoch + 1 for m in training_metrics]
    val_losses = [m.val_loss for m in training_metrics]

    @themed
    def plot():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.plot(epochs, val_losses)
        return fig

    mo.Html(plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generate continuations

    Inference runs through the apparatus too: we don't need to download the
    (potentially large) model. Only the results come back.
    """)
    return


@app.function(hide_code=True)
def generate(prompts: list[str], max_new_tokens: int, temperature: float):
    """Load the trained model and generate continuations."""
    from typing import cast

    import jax.random as jr
    import numpy as np

    from experiment.compute.model import load_checkpoint
    from experiment.data.tokenizer import CharTokenizer

    data_dir = get_data_dir()
    log.info("Loading model from checkpoint")
    model, cfg, _ = load_checkpoint(data_dir)
    tokenizer = CharTokenizer(cfg.tokenizer)
    context = np.asarray(tokenizer.encode(prompts, cfg.model.block_size), dtype=np.int32)

    log.info(f"Generating {max_new_tokens} tokens at temperature {temperature}")
    output = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, key=jr.key(cfg.seed))

    toks = cast(list[list[int]], output.tokens.tolist())
    return tokenizer.decode_each(toks), output


@app.cell(hide_code=True)
async def _(app):
    prompts = [
        "It is a truth uni",
        "Mr. Darcy walked across the",
    ]
    continuations, gen_metadata = await app.arun(
        partial(generate, prompts=prompts, max_new_tokens=300, temperature=0.5),
    )

    for seq in continuations:
        print("".join(seq)[:80])
    return continuations, gen_metadata


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Token metrics: Surprisal and entropy

    * **Entropy** measures how diffuse the next-token distribution is *before*
      sampling -- the model's uncertainty.
    * **Surprisal** measures how unlikely the chosen token was -- the
      cross-entropy loss for that position.

    Together they reveal how the prompt and temperature affect generation.
    Notably, *entropy is unaffected by temperature* whereas surprisal *is*
    (because it's calculated after sampling).
    """)
    return


@app.cell(hide_code=True)
def _(continuations, gen_metadata):
    from subline.series import Series
    from subline.subline import Subline

    viz = Subline(chars_per_line=80)
    svg = viz.plot(
        continuations[0],
        [
            Series(gen_metadata[0].surprise_surprise, label="S\u2082"),
            Series(-gen_metadata[0].surprise_surprise, label="-S\u2082", dasharray="1"),
        ],
    )
    mo.Html(svg)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## References

    Karpathy, A. (2022). nanoGPT [Computer software]. GitHub.
    https://github.com/karpathy/nanoGPT

    Sanderson, G. (2024a). Visualizing attention, a transformer's heart.
    3Blue1Brown. https://www.3blue1brown.com/lessons/attention

    Sanderson, G. (2024b). How might LLMs store facts. 3Blue1Brown.
    https://www.3blue1brown.com/lessons/mlp
    """)
    return


@app.cell(hide_code=True)
def options():
    app_type = mo.ui.radio(
        label="Apparatus",
        options=["local", "modal"],
        value=str(mo.cli_args().get("app", "local")),
        inline=True,
    )
    arch = mo.ui.radio(
        label="Architecture",
        options=["gpt", "ngpt"],
        value=str(mo.cli_args().get("arch", "gpt")),
        inline=True,
    )
    ngpt_variant = mo.ui.radio(
        label="nGPT variant",
        options=["crude", "full"],
        value=str(mo.cli_args().get("ngpt_variant", "crude")),
        inline=True,
    )
    run_button = mo.ui.run_button(
        label="Run",
    )
    is_headless = mo.app_meta().request is None
    return app_type, arch, is_headless, ngpt_variant, run_button


if __name__ == "__main__":
    app.run()

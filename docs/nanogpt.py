import marimo

__generated_with = '0.20.1'
app = marimo.App(width='medium')

with app.setup(hide_code=True):
    import marimo as mo  # noqa: F401
    import logging
    from functools import partial

    from mini import LocalApparatus, ModalApparatus  # noqa: F401
    from mini import get_data_dir
    from mini.logging import SimpleLoggingConfig
    from mini.vis import themed
    from utils.lr_finder.vis import plot_lr_finder

    logging_config = SimpleLoggingConfig().info('notebook', 'experiment', 'mini', 'utils')
    logging_config.apply()

    log = logging.getLogger('notebook')


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Character-level nanoGPT

    This experiment trains a tiny transformer on character-level data, based on a port
    of [nanoGPT](https://github.com/karpathy/nanoGPT). Most of the code lives in
    modules under [src/experiment](../src/experiment); this notebook ties it together.
    """)
    return


@app.cell
def _():
    from experiment.config import (
        DataConfig,
        MixedPrecisionConfig,
        ModelConfig,
        OptimizerConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )

    config = TrainingConfig(
        model=ModelConfig(
            vocab_size=64,  # set after loading the dataset
            block_size=512,
            n_embd=32,
            n_head=8,
            n_head_dim=8,
            n_ff=128,
            n_layer=12,
            dropout=0.1,
        ),
        tokenizer=TokenizerConfig(vocabulary=[]),
        data=DataConfig(
            batch_size=16,
            oversample=1,
            train_split=0.8,
            padding_chance=0.1,
        ),
        optimizer=OptimizerConfig(
            weight_decay=1e-3,
            learning_rate=0,  # set by LR finder
            betas=(0.9, 0.95),
        ),
        scheduler=SchedulerConfig(
            epochs=100,
            warmup_epochs=10,
            min_lr_factor=0.01,
        ),
        amp=MixedPrecisionConfig(enabled=False),
    )
    return (config,)


@app.cell(hide_code=True)
def _(app_type):
    mo.md(f"""
    {app_type}
    """)
    return


@app.cell(hide_code=True)
def _(app_type):
    if app_type.value == 'local':
        app = LocalApparatus('nanogpt')
    else:
        app = ModalApparatus('nanogpt').w(gpu='L4', max_containers=1).before_each(logging_config.apply)
    mo.output.append(mo.md(f'Using **{app}**'))
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
    x = self.data[idx : idx + self.block_size]
    y = self.data[idx + 1 : idx + self.block_size + 1]
    ```
    """)
    return


@app.function
def download_pride_and_prejudice():
    """Download Pride and Prejudice from the Gutenberg HuggingFace dataset."""
    from experiment.config import DatasetMetadata

    import ftfy
    import pandas as pd

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


@app.function
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
def _(app, config):
    from experiment.utils import align

    input_metadata = app.run(prepare_data)

    config.tokenizer = input_metadata.tokenizer_config.model_copy()
    config.model.vocab_size = align(config.tokenizer.vocab_size, 64)

    input_metadata.model_dump(exclude={'tokenizer_config'})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Learning rate search

    Before training, we run a multi-scale learning rate range test. The finder
    progressively narrows the search space to improve stability.
    """)
    return


@app.function
def find_learning_rate(config):
    """Run a multi-scale LR range test and return (lr, config, history)."""
    import torch

    from experiment.compute.data_pipelines import load_data
    from experiment.data.dataloader import get_dataloader
    from experiment.model.gpt import GPT
    from experiment.training.optimizer import configure_optimizer
    from utils.lr_finder.lr_finder import lr_finder_search
    from utils.torch.mixed_precision import AMPContext
    from utils.torch.types import get_device

    data_dir = get_data_dir()
    model = GPT(config.model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = configure_optimizer(model, config.optimizer)
    data, _ = load_data(data_dir)

    if torch.cuda.is_available():
        data = data.cuda()
        model = model.cuda()
        criterion = criterion.cuda()

    train_loader, _ = get_dataloader(data, config.data, config.model)
    amp_context = AMPContext(use_amp=config.amp.enabled, device_type=get_device(model), dtype=config.amp.dtype)

    return lr_finder_search(model, criterion, optimizer, train_loader, amp_context=amp_context)


@app.cell
async def _(app, config):
    suggested_lr, lr_config, lr_history = await app.arun(find_learning_rate, config)

    config.optimizer.learning_rate = suggested_lr
    mo.output.append(mo.md(f'Suggested learning rate: **{suggested_lr:.2e}**'))
    return lr_config, lr_history


@app.cell
def _(lr_config, lr_history):
    mo.Html(
        themed(plot_lr_finder, alt_text='Learning-rate finder plot')(
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


@app.function
def train(config):
    """Run a full training loop. Return per-epoch metrics."""
    from experiment.compute.training import train_model

    data_dir = get_data_dir()
    _, metrics = train_model(config, data_dir)
    return metrics


@app.cell
async def _(app, config):
    import matplotlib.pyplot as plt

    mo.stop(True)
    training_metrics = await app.arun(train, config)

    # Plot training curve
    epochs = [m.epoch + 1 for m in training_metrics]
    val_losses = [m.val_loss for m in training_metrics]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title('Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(epochs, val_losses)
    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generate continuations

    Inference runs through the apparatus too: we don't need to download the
    (potentially large) model. Only the results come back.
    """)
    return


@app.function
def generate(prompts: list[str], max_new_tokens: int, temperature: float):
    """Load the trained model and generate continuations."""
    from typing import cast

    import torch

    from experiment.compute.model import load_checkpoint
    from experiment.data.tokenizer import CharTokenizer

    data_dir = get_data_dir()
    log.info('Loading model from checkpoint')
    model, cfg, _ = load_checkpoint(data_dir)
    model.eval()
    tokenizer = CharTokenizer(cfg.tokenizer)
    context = torch.tensor(tokenizer.encode(prompts, cfg.model.block_size), dtype=torch.long)
    if torch.cuda.is_available():
        context = context.cuda()
        model = model.cuda()

    log.info(f'Generating {max_new_tokens} tokens at temperature {temperature}')
    output = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature)

    toks = cast(list[list[int]], output.tokens.tolist())
    return tokenizer.decode_each(toks), output


@app.cell
def _(app):
    prompts = [
        'It is a truth uni',
        'Mr. Darcy walked across the',
    ]
    mo.stop(True)
    continuations, gen_metadata = app.run(
        partial(generate, prompts=prompts, max_new_tokens=300, temperature=0.5),
    )

    for seq in continuations:
        print(''.join(seq)[:80])
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


@app.cell
def _(continuations, gen_metadata):
    from subline.series import Series
    from subline.subline import Subline

    viz = Subline(chars_per_line=80)
    svg = viz.plot(
        continuations[0],
        [
            Series(gen_metadata[0].surprise_surprise, label='S\u2082'),
            Series(-gen_metadata[0].surprise_surprise, label='-S\u2082', dasharray='1'),
        ],
    )
    mo.Html(svg)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Future research

    ### Temperature as a resource

    The first character of each word tends to have high entropy and high surprisal,
    with subsequent characters lower. Where the model makes spelling mistakes it
    often had low entropy but then high surprisal -- suggesting it knows what it
    wants to write but sampling messes it up.

    An idea: lower the temperature after the first letter of each word. How would
    that generalise to languages without spaces? Perhaps temperature could be a
    resource that gets consumed (by picking unlikely tokens) and gradually
    replenished (by picking likely ones).
    """)
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

    ## Software Licenses

    The code in this notebook is derived from nanoGPT (Karpathy, 2022), which is
    licensed under the MIT License, Copyright (c) 2022 Andrej Karpathy.
    """)
    return


@app.cell(hide_code=True)
def _():
    app_type = mo.ui.dropdown(
        label='Apparatus',
        options=['local', 'modal'],
        value=mo.cli_args().get('app', 'modal'),
    )
    return (app_type,)


if __name__ == '__main__':
    app.run()

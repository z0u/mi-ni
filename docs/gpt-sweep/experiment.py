"""
Architecture sweep: GPT versus nGPT, as a memoized experiment.

A controlled comparison of the baseline LayerNorm GPT against nGPT, swept across
three peak learning rates (3 architectures × 3 LRs = 9 training runs). One CPU-ish
data-prep step, then a GPU sweep whose configs depend on prep's tokenizer.

This is the *definition* — an importable ``main(ctx)`` DAG with no compute baked
in. Drive it on Modal L4s from the CLI; the companion ``report.py`` reads the
durable results and renders them.

    # one data-prep run, then nine training runs, fanned out across L4s:
    bin/mini run docs/gpt-sweep/experiment.py --app modal --max-containers 9
    bin/mini status gpt-sweep    # no --app needed — the launch backend sticks

The hardware is bound by role (see ``roles`` below): ``prep`` runs CPU-only, ``train``
on L4s — so ``main`` names labels, not GPUs. Re-run to advance/resume — done cells are memo hits, so a crash heals by re-running
and a failed cell is recovered with ``bin/mini retry gpt-sweep``. Adding an LR or
architecture below and re-running launches only the new cells.
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir

# Axes of the sweep.
LRS = [("3e-3", 3e-3), ("1e-2", 1e-2), ("4e-2", 4e-2)]
ARCH_CFGS = [
    ("baseline", dict(architecture="gpt")),
    ("nGPT", dict(architecture="ngpt", ngpt_variant="full")),
    ("nGPT (scalar)", dict(architecture="ngpt", ngpt_variant="crude")),
]

# Named view of the gathered val-loss curves in the project-scoped store. The
# report resolves this ref at export time, so the data lives in the durable store
# (the HF bucket when configured), not committed to Git.
CURVES_REF = "reports/gpt-sweep/curves"


def download_pride_and_prejudice():
    """Download Pride and Prejudice from the Gutenberg HuggingFace dataset."""
    import ftfy
    import pandas as pd

    from experiment.config import DatasetMetadata

    url = "https://huggingface.co/api/datasets/larenwell/book-gutenberg-train/parquet/default/train/0.parquet"
    df = pd.read_parquet(url, columns=["text"])
    text = df.iloc[0]["text"]
    text, explanation = ftfy.fix_and_explain(text)
    return text, DatasetMetadata(
        title="Pride and Prejudice",
        author="Jane Austen",
        url=url,
        fixes=explanation or [],
        total_chars=len(text),
    )


def prepare_data():
    """Download, tokenize, and save training data to the volume; return the corpus metadata."""
    from experiment.compute.data_pipelines import save_data
    from experiment.data.preparation import tokenize_data

    data_dir = get_data_dir()
    data, metadata = tokenize_data([download_pride_and_prejudice()])
    save_data(data, metadata, data_dir)
    return metadata


def _make_config(lr_float: float, arch_kwargs: dict):
    """Build one training config (vocab/tokenizer filled in after prep)."""
    from experiment.config import (
        DataConfig,
        ModelConfig,
        OptimizerConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )

    is_ngpt = arch_kwargs.get("architecture") == "ngpt"
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
        data=DataConfig(batch_size=16, oversample=2, train_split=0.8, padding_chance=0.1),
        optimizer=OptimizerConfig(
            weight_decay=0 if is_ngpt else 1e-3,
            learning_rate=lr_float,
            betas=(0.9, 0.95),
        ),
        scheduler=SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01),
    )


def build_sweep(meta) -> list[tuple]:
    """Derive the (config, arch_label, lr_str) cells from prep's tokenizer.

    Runs every wake (cheap + deterministic), so the memo keys are stable: each
    cell re-runs only if its own config changes.
    """
    from experiment.utils import align

    cells = []
    for lr_str, lr_float in LRS:
        for arch_label, arch_kwargs in ARCH_CFGS:
            config = _make_config(lr_float, arch_kwargs)
            config.tokenizer = meta.tokenizer_config.model_copy()
            config.model.vocab_size = align(meta.tokenizer_config.vocab_size, 64)
            cells.append((config, arch_label, lr_str))
    return cells


def train_one(config, arch_label: str, lr_str: str) -> tuple:
    """Train one sweep cell; return its arch label, LR string, and per-epoch val losses."""
    from experiment.compute.training import train_model

    _, metrics = train_model(config, get_data_dir())
    return arch_label, lr_str, [m.val_loss for m in metrics]


def publish_curves(results: list[tuple]) -> str:
    """Publish the gathered val-loss curves to the project store under ``CURVES_REF``.

    A step, so the worker binds the ambient store and bare ``put`` / ``set_ref``
    resolve against it (the HF bucket when configured — the token rides in on the
    worker's Secret). The report then reads the curves by name, so the data lives in
    the durable store rather than a ``results.json`` in Git. Idempotent: ``put`` is
    content-addressed, and ``set_ref`` is fenced on the attempt generation — only
    the current attempt can move the name; a stale relaunch fails loudly instead.
    """
    import json

    from mini.store import put, set_ref

    curves = {f"{arch}|{lr}": losses for arch, lr, losses in results}
    set_ref(CURVES_REF, put(json.dumps(curves, indent=2).encode(), name="gpt-sweep-curves.json"))
    return CURVES_REF


def main(ctx: Ctx) -> list[tuple]:
    meta = ctx.run(prepare_data, role="prep")  # CPU prep; suspends until done
    configs, archs, lrs = zip(*build_sweep(meta), strict=True)
    results = ctx.map(train_one, configs, archs, lrs, role="train")  # GPU sweep that depends on prep
    ctx.run(publish_curves, results, role="prep")  # share the curves by name for the report
    return results


experiment = Experiment(
    name="gpt-sweep",
    main=main,
    roles={
        "prep": {},  # CPU-only: data download + tokenize
        "train": dict(gpu="L4", timeout=720),  # GPU sweep cells
    },
)

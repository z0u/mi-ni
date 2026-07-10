"""
Architecture sweep: GPT versus nGPT, plus the nGPT residual-bug ablation.

Two coordinated sweeps over one data-prep step, run on Modal L4s:

1. **Architecture × learning rate** (9 cells) at the base size — baseline
   LayerNorm GPT against nGPT (per-channel and scalar gates), swept across three
   peak LRs. The "does normalization help?" comparison.
2. **Residual form × width** (9 cells) — the scalar-gate nGPT across widths {32,
   64, 128} with ``n_ff = 4·n_embd``, in three residual recipes: the correct
   normalized-LERP step (``norm``, the fix and default); the raw additive step
   with mi-ni's learnable, width-scaled gate (``add``); and the additive step
   with a *fixed* gate ``α = 1/n_layer`` (``fixed``). The additive step's
   effective rotation rides on ``‖sublayer‖ ∝ √n_embd``, so ``fixed`` diverges
   wide; the learnable gate adapts with width and masks it, and the normalized
   LERP is width-flat by construction.

This is the *definition* — an importable ``main(ctx)`` DAG with no compute baked
in. Drive it on Modal L4s from the CLI; the companion ``report.py`` reads the
durable results and renders them.

    # one data-prep run, then eighteen training runs fanned out across L4s:
    bin/mini run docs/gpt-sweep/experiment.py --app modal --max-containers 18
    bin/mini status gpt-sweep    # no --app needed — the launch backend sticks

The hardware is bound by role (see ``roles`` below): ``prep`` runs CPU-only, ``train``
on L4s — so ``main`` names labels, not GPUs. Re-run to advance/resume — done cells are memo hits, so a crash heals by re-running
and a failed cell is recovered with ``bin/mini retry gpt-sweep``. Adding an LR,
architecture, or width below and re-running launches only the new cells.
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir

# Sweep 1 axes: architecture × peak learning rate, at the base size.
LRS = [("3e-3", 3e-3), ("1e-2", 1e-2), ("4e-2", 4e-2)]
ARCH_CFGS = [
    ("baseline", dict(architecture="gpt")),
    ("nGPT", dict(architecture="ngpt", ngpt_variant="full")),
    ("nGPT (scalar)", dict(architecture="ngpt", ngpt_variant="crude")),
]

# Sweep 2 axes: the residual-bug ablation. The scalar-gate nGPT, normalized-LERP
# residual (the fix) vs the raw additive step (the bug), across widths — held at
# the LR where nGPT is happiest (1e-2) and the base depth (12). n_ff scales with
# width so the MLP output norm (∝ √n_embd) actually grows, which is what the
# additive step fails to control.
WIDTH_LR = 1e-2
WIDTHS = [32, 64, 128]
# (label, model-config knobs): the normalized-LERP fix (default), the additive
# step with mi-ni's learnable width-scaled gate, and the additive step with a
# fixed gate — the last is what exposes the width-gated failure.
RESID_FORMS = [
    ("norm", dict(normalize_sublayer=True)),
    ("add", dict(normalize_sublayer=False)),
    ("fixed", dict(normalize_sublayer=False, learnable_alpha=False)),
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


def _make_config(lr_float: float, arch_kwargs: dict, *, n_embd: int = 32, n_ff: int = 128, n_layer: int = 12):
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
            n_embd=n_embd,
            n_head=8,
            n_head_dim=8,
            n_ff=n_ff,
            n_layer=n_layer,
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
    """Derive the (config, label) cells for both sweeps from prep's tokenizer.

    Runs every wake (cheap + deterministic), so the memo keys are stable: each
    cell re-runs only if its own config changes.
    """
    from experiment.utils import align

    cells: list[tuple] = []
    # Sweep 1: architecture × LR at the base size.
    for lr_str, lr_float in LRS:
        for arch_label, arch_kwargs in ARCH_CFGS:
            cells.append((_make_config(lr_float, arch_kwargs), f"{arch_label}|{lr_str}"))
    # Sweep 2: residual form × width (scalar-gate nGPT), n_ff = 4·n_embd.
    for n_embd in WIDTHS:
        for form_label, resid_kwargs in RESID_FORMS:
            arch_kwargs = dict(architecture="ngpt", ngpt_variant="crude", **resid_kwargs)
            config = _make_config(WIDTH_LR, arch_kwargs, n_embd=n_embd, n_ff=4 * n_embd)
            cells.append((config, f"width|{form_label}|{n_embd}"))

    for config, _label in cells:
        config.tokenizer = meta.tokenizer_config.model_copy()
        config.model.vocab_size = align(meta.tokenizer_config.vocab_size, 64)
    return cells


def train_one(config, label: str) -> tuple:
    """Train one sweep cell; return its label and per-epoch val losses."""
    from experiment.compute.training import train_model

    _, metrics = train_model(config, get_data_dir())
    return label, [m.val_loss for m in metrics]


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

    curves = dict(results)
    set_ref(CURVES_REF, put(json.dumps(curves, indent=2).encode(), name="gpt-sweep-curves.json"))
    return CURVES_REF


def main(ctx: Ctx) -> list[tuple]:
    meta = ctx.run(prepare_data, role="prep")  # CPU prep; suspends until done
    configs, labels = zip(*build_sweep(meta), strict=True)
    results = ctx.map(train_one, configs, labels, role="train")  # GPU sweep that depends on prep
    ctx.run(publish_curves, results, role="prep")  # share the curves by name for the report
    return results


experiment = Experiment(
    name="gpt-sweep",
    main=main,
    roles={
        "prep": {},  # CPU-only: data download + tokenize
        # L4 is right-sized for these batch-16 cells; the largest (width 128,
        # depth 12) takes ~13 min, so the per-task timeout allows generous slack.
        "train": dict(gpu="L4", timeout=1500),  # GPU sweep cells
    },
)

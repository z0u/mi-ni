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

The hardware is bound by role (see ``roles`` below): ``prep`` runs CPU-only, ``train``
on L4s — so ``main`` names labels, not GPUs. Re-run to advance/resume — done cells are memo hits, so a crash heals by re-running
and a failed cell is recovered with ``bin/mini retry gpt-sweep``. Adding an LR or
architecture below and re-running launches only the new cells.
"""

from __future__ import annotations

from experiment.corpus import prepare_data
from mini import Ctx, Experiment, get_data_dir

# Axes of the sweep.
LRS = [('3e-3', 3e-3), ('1e-2', 1e-2), ('4e-2', 4e-2)]
ARCH_CFGS = [
    ('baseline', dict(architecture='gpt')),
    ('nGPT', dict(architecture='ngpt', ngpt_variant='full')),
    ('nGPT (scalar)', dict(architecture='ngpt', ngpt_variant='crude')),
]


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


def main(ctx: Ctx) -> list[tuple]:
    meta = ctx.run(prepare_data, role='prep')  # CPU prep; suspends until done
    return ctx.map(train_one, build_sweep(meta), role='train')  # GPU sweep that depends on prep


experiment = Experiment(
    name='gpt-sweep',
    main=main,
    roles={
        'prep': {},  # CPU-only: data download + tokenize
        'train': dict(gpu='L4', timeout=720),  # GPU sweep cells
    },
)

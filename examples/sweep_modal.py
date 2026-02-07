#!/usr/bin/env python3
"""
Example: run a hyperparameter sweep on Modal.

Same experiment as sweep_local.py, but scaled up to run on Modal with
optional GPU support.  This demonstrates switching from local to remote
execution by changing only the executor.

Usage::

    uv run python examples/sweep_modal.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import modal

from mini.executor import ModalExecutor

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App('executor-sweep-example')

image = (
    modal.Image.debian_slim(python_version='3.13')
    .pip_install('torch', 'pydantic', 'jaxtyping', 'ftfy', 'numpy', 'scikit-learn')
    .add_local_python_source('experiment', 'mini', 'utils')
)


# ---------------------------------------------------------------------------
# Train function — identical logic to sweep_local.py, but the data and
# model setup happens inside the remote worker.
# ---------------------------------------------------------------------------


@dataclass
class TrainParams:
    seed: int
    lr: float
    epochs: int = 20


@dataclass
class TrainResult:
    seed: int
    lr: float
    final_val_loss: float
    epochs_trained: int


SAMPLE_TEXT = (
    'To be, or not to be, that is the question: '
    'Whether tis nobler in the mind to suffer '
    'The slings and arrows of outrageous fortune, '
    'Or to take arms against a sea of troubles, '
    'And by opposing end them. To die, to sleep; '
    'No more; and by a sleep to say we end '
    'The heart-ache and the thousand natural shocks '
    'That flesh is heir to — tis a consummation '
    'Devoutly to be wished. To die, to sleep; '
    'To sleep, perchance to dream — ay, theres the rub: '
    'For in that sleep of death what dreams may come, '
    'When we have shuffled off this mortal coil, '
    'Must give us pause.'
) * 20


def train(params: TrainParams) -> TrainResult:
    """Train a tiny GPT model.  Runs inside a Modal container."""
    import torch
    from torch.utils.data import DataLoader

    from experiment.config import (
        DataConfig,
        MixedPrecisionConfig,
        ModelConfig,
        OptimizerConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )
    from experiment.data.dataset import OverlappingRandomSampler, TextDataset
    from experiment.data.tokenizer import CharTokenizer
    from experiment.model.gpt import GPT
    from experiment.training.metrics import TrainingMetrics
    from experiment.training.optimizer import configure_optimizer
    from experiment.training.scheduler import configure_scheduler
    from utils.torch.mixed_precision import AMPContext
    from utils.torch.types import get_device

    torch.manual_seed(params.seed)

    vocabulary = sorted(set(SAMPLE_TEXT))
    config = TrainingConfig(
        model=ModelConfig(
            vocab_size=64,
            block_size=64,
            n_embd=64,
            n_head=8,
            n_head_dim=8,
            n_ff=128,
            n_layer=2,
            dropout=0.0,
        ),
        tokenizer=TokenizerConfig(vocabulary=vocabulary),
        data=DataConfig(batch_size=8, oversample=1, train_split=0.8, padding_chance=0.0),
        optimizer=OptimizerConfig(weight_decay=0.0, learning_rate=params.lr, betas=(0.9, 0.95)),
        scheduler=SchedulerConfig(epochs=params.epochs, warmup_epochs=2, min_lr_factor=0.01),
        amp=MixedPrecisionConfig(enabled=False),
    )

    tokenizer = CharTokenizer(config.tokenizer)
    tokens = tokenizer.encode([SAMPLE_TEXT])[0]
    data = torch.tensor(tokens, dtype=torch.long)

    model = GPT(config.model)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    if torch.cuda.is_available():
        data, model, criterion = data.cuda(), model.cuda(), criterion.cuda()

    n = int(config.data.train_split * len(data))
    train_ds = TextDataset(data[:n], config.model.block_size, config.data.padding_chance)
    val_ds = TextDataset(data[n:], config.model.block_size, config.data.padding_chance)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        sampler=OverlappingRandomSampler(train_ds, config.data.batch_size, config.model.block_size, oversample=2),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        sampler=OverlappingRandomSampler(val_ds, config.data.batch_size, config.model.block_size),
    )

    optimizer = configure_optimizer(model, config.optimizer)
    scheduler = configure_scheduler(optimizer, config.scheduler, epoch_length=len(train_loader))
    amp_ctx = AMPContext(use_amp=config.amp.enabled, device_type=get_device(model), dtype=config.amp.dtype)

    last_metrics = None
    for epoch in range(config.scheduler.epochs):
        model.train()
        for xb, yb in train_loader:
            with amp_ctx.forward_pass():
                logits = model(xb)
                loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            amp_ctx.backward_pass(loss, optimizer)
            scheduler.step()

        total_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                with amp_ctx.forward_pass():
                    logits = model(xb)
                    loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        last_metrics = TrainingMetrics(
            epoch=epoch,
            learning_rate=scheduler.get_last_lr()[0],
            val_loss=val_loss,
            training_tokens=(epoch + 1) * len(train_loader) * config.data.batch_size * config.model.block_size,
        )
        # This print is streamed back by Modal and visible locally
        print(f'[seed={params.seed} lr={params.lr:.1e}] epoch {epoch}: val_loss={val_loss:.4f}')

    assert last_metrics is not None
    return TrainResult(
        seed=params.seed,
        lr=params.lr,
        final_val_loss=last_metrics.val_loss,
        epochs_trained=config.scheduler.epochs,
    )


# ---------------------------------------------------------------------------
# Run the sweep
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    executor = ModalExecutor(app).with_modal_kwargs(
        image=image,
        # gpu=None,       # CPU-only (cheapest)
        # gpu='T4',       # budget GPU
        timeout=600,
    )

    sweep = [
        TrainParams(seed=0, lr=1e-2),
        TrainParams(seed=0, lr=3e-3),
        TrainParams(seed=0, lr=1e-3),
        TrainParams(seed=0, lr=3e-4),
        TrainParams(seed=1, lr=1e-2),
        TrainParams(seed=1, lr=3e-3),
    ]

    print(f'Running {len(sweep)} training jobs on Modal...\n')
    results = list(executor.map(train, sweep))

    print('\nResults:')
    print(f'  {"seed":>6s}  {"lr":>10s}  {"val_loss":>10s}')
    print(f'  {"—" * 6}  {"—" * 10}  {"—" * 10}')
    for r in results:
        print(f'  {r.seed:6d}  {r.lr:10.1e}  {r.final_val_loss:10.4f}')

    best = min(results, key=lambda r: r.final_val_loss)
    print(f'\nBest: seed={best.seed}, lr={best.lr:.1e}, val_loss={best.final_val_loss:.4f}')

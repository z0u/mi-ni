#!/usr/bin/env python3
"""
Example: run a small hyperparameter sweep locally.

This demonstrates the executor pattern with a tiny GPT model trained on
synthetic data. No GPU or Modal account needed --- it runs entirely on CPU.

Usage::

    uv run python examples/sweep_local.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
from mini.executor import LocalExecutor, get_progress
from utils.logging import SimpleLoggingConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
SimpleLoggingConfig().info('mini').apply()


# ---------------------------------------------------------------------------
# Tiny config for fast local iteration
# ---------------------------------------------------------------------------

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
) * 20  # Repeat to have enough data


def make_config(lr: float, seed: int, epochs: int = 5) -> TrainingConfig:
    vocabulary = sorted(set(SAMPLE_TEXT))
    return TrainingConfig(
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
        data=DataConfig(
            batch_size=8,
            oversample=1,
            train_split=0.8,
            padding_chance=0.0,
        ),
        optimizer=OptimizerConfig(
            weight_decay=0.0,
            learning_rate=lr,
            betas=(0.9, 0.95),
        ),
        scheduler=SchedulerConfig(
            epochs=epochs,
            warmup_epochs=1,
            min_lr_factor=0.01,
        ),
        amp=MixedPrecisionConfig(enabled=False),
    )


# ---------------------------------------------------------------------------
# Self-contained train function (no filesystem dependency)
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    seed: int
    lr: float
    final_val_loss: float
    epochs_trained: int


@dataclass
class TrainParams:
    seed: int
    lr: float
    epochs: int = 5


def train(params: TrainParams) -> TrainResult:
    """
    Train a tiny GPT and return the final metrics.

    Reports progress via mini.executor.get_progress() if available.
    """
    torch.manual_seed(params.seed)
    config = make_config(lr=params.lr, seed=params.seed, epochs=params.epochs)

    # Tokenize in memory
    tokenizer = CharTokenizer(config.tokenizer)
    tokens = tokenizer.encode([SAMPLE_TEXT])[0]
    data = torch.tensor(tokens, dtype=torch.long)

    # Model
    model = GPT(config.model)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    # Data loaders
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

    # Progress reporting
    progress = get_progress()
    total_steps = config.scheduler.epochs * (len(train_loader) + len(val_loader))
    if progress:
        progress.set_total(total_steps)

    # Training loop
    last_metrics = None
    for epoch in range(config.scheduler.epochs):
        model.train()
        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if progress:
                progress.update(1)

        # Validation
        total_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                total_val_loss += loss.item()
                if progress:
                    progress.update(1)

        val_loss = total_val_loss / len(val_loader)
        last_metrics = TrainingMetrics(
            epoch=epoch,
            learning_rate=scheduler.get_last_lr()[0],
            val_loss=val_loss,
            training_tokens=(epoch + 1) * len(train_loader) * config.data.batch_size * config.model.block_size,
        )
        if progress:
            progress.set_message(f'epoch {epoch} val_loss={val_loss:.3f}')

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
    executor = LocalExecutor('lr-sweep', max_workers=2)

    sweep = [
        TrainParams(seed=0, lr=1e-2),
        TrainParams(seed=0, lr=3e-3),
        TrainParams(seed=0, lr=1e-3),
        TrainParams(seed=0, lr=3e-4),
    ]

    print(f'Running {len(sweep)} training jobs...\n')
    results = list(executor.map(train, sweep))

    print('\nResults:')
    print(f'  {"lr":>10s}  {"val_loss":>10s}')
    print(f'  {"—" * 10}  {"—" * 10}')
    for r in results:
        print(f'  {r.lr:10.1e}  {r.final_val_loss:10.4f}')

    best = min(results, key=lambda r: r.final_val_loss)
    print(f'\nBest: lr={best.lr:.1e}, val_loss={best.final_val_loss:.4f}')

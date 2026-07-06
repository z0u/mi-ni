from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiment.compute.data_pipelines import load_data
from experiment.compute.model import save_checkpoint
from experiment.config import TrainingConfig
from experiment.data.batches import batches_per_epoch, sample_batches, split_data
from experiment.model import LanguageModel, build_model
from experiment.training.loop import eval_step, make_train_step
from experiment.training.metrics import TrainingMetrics
from experiment.training.optimizer import configure_optimizer
from experiment.training.scheduler import configure_schedule
from mini.progress import emit_progress


def train_model(
    config: TrainingConfig,
    data_dir: Path,
    checkpoint_every: int | None = None,
) -> tuple[LanguageModel, list[TrainingMetrics]]:
    """Train a model and return it with per-epoch metrics.

    Args:
        config: Full training configuration.
        data_dir: Directory for loading data and saving checkpoints.
        checkpoint_every: Save a checkpoint every N epochs. None = only at the end.
    """
    data, metadata = load_data(data_dir)
    assert metadata.tokenizer_config.vocab_size <= config.model.vocab_size, "Vocab size mismatch"

    model_key, dropout_key = jr.split(jr.key(config.seed))
    model = build_model(config.model, key=model_key)
    rng = np.random.default_rng(config.seed)

    train_data, val_data = split_data(data, config.data.train_split)
    epoch_length = batches_per_epoch(len(train_data), config.data, config.model)
    val_length = batches_per_epoch(len(val_data), config.data, config.model, oversample=1)

    if checkpoint_every is None:
        checkpoint_every = max(1, config.scheduler.epochs // 50)

    schedule = configure_schedule(config.scheduler, config.optimizer.learning_rate, epoch_length)
    optimizer = configure_optimizer(model, config.optimizer, schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_train_step(optimizer)

    total_steps = config.scheduler.epochs * epoch_length
    tokens_per_epoch = epoch_length * config.data.batch_size * config.model.block_size
    all_metrics: list[TrainingMetrics] = []
    step = 0

    for epoch in range(config.scheduler.epochs):
        for x, y in sample_batches(train_data, config.data, config.model, epoch_length, rng):
            dropout_key, step_key = jr.split(dropout_key)
            model, opt_state, loss = train_step(model, opt_state, x, y, step_key)
            step += 1
            emit_progress(step, total_steps, message=f"loss={float(loss):.4f}")

        val_losses = [
            float(eval_step(model, x, y))
            for x, y in sample_batches(val_data, config.data, config.model, val_length, rng)
        ]
        metrics = TrainingMetrics(
            epoch=epoch,
            learning_rate=float(jnp.asarray(schedule(step))),
            val_loss=float(np.mean(val_losses)),
            training_tokens=(epoch + 1) * tokens_per_epoch,
        )
        all_metrics.append(metrics)

        if epoch > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(model, config, metrics, data_dir)

    if all_metrics:
        save_checkpoint(model, config, all_metrics[-1], data_dir)

    return model, all_metrics

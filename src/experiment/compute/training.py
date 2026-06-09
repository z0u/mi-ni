from pathlib import Path
from typing import Literal

import lightning as L

from experiment.compute.data_pipelines import load_data
from experiment.compute.model import save_checkpoint
from experiment.config import MixedPrecisionConfig, TrainingConfig
from experiment.data.dataloader import get_dataloader
from experiment.model import LanguageModel, build_model
from experiment.training.metrics import TrainingMetrics
from experiment.training.module import GPTModule
from mini.torch.lightning import LightningProgress


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
    assert metadata.tokenizer_config.vocab_size <= config.model.vocab_size, 'Vocab size mismatch'

    model = build_model(config.model)
    train_loader, val_loader = get_dataloader(data, config.data, config.model)

    if checkpoint_every is None:
        checkpoint_every = max(1, config.scheduler.epochs // 50)

    tokens_per_epoch = len(train_loader) * config.data.batch_size * config.model.block_size
    metrics_cb = _MetricsCallback(checkpoint_every, config, model, data_dir, tokens_per_epoch)

    trainer = L.Trainer(
        max_epochs=config.scheduler.epochs,
        precision=_precision(config.amp),
        callbacks=[LightningProgress(), metrics_cb],
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=min(50, len(train_loader)),
    )
    trainer.fit(
        GPTModule(model, config, epoch_length=len(train_loader)),
        train_loader,
        val_loader,
    )

    return model, metrics_cb.all_metrics


def _precision(amp: MixedPrecisionConfig) -> Literal['32-true', '16-mixed', 'bf16-mixed']:
    """Map MixedPrecisionConfig to a Lightning precision string."""
    if not amp.enabled:
        return '32-true'
    dtype = amp.dtype or 'float16'
    return 'bf16-mixed' if dtype == 'bfloat16' else '16-mixed'


class _MetricsCallback(L.Callback):
    """Collect per-epoch metrics and save checkpoints on schedule."""

    def __init__(
        self,
        checkpoint_every: int,
        config: TrainingConfig,
        model: LanguageModel,
        data_dir: Path,
        tokens_per_epoch: int,
    ):
        self.checkpoint_every = checkpoint_every
        self.config = config
        self.model = model
        self.data_dir = data_dir
        self.tokens_per_epoch = tokens_per_epoch
        self.all_metrics: list[TrainingMetrics] = []

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        epoch = trainer.current_epoch
        val_loss = float(trainer.callback_metrics.get('val_loss', float('nan')))
        lr = float(trainer.optimizers[0].param_groups[0]['lr'])
        metrics = TrainingMetrics(
            epoch=epoch,
            learning_rate=lr,
            val_loss=val_loss,
            training_tokens=(epoch + 1) * self.tokens_per_epoch,
        )
        self.all_metrics.append(metrics)

        if epoch > 0 and epoch % self.checkpoint_every == 0:
            save_checkpoint(self.model, self.config, metrics, self.data_dir)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self.all_metrics:
            save_checkpoint(self.model, self.config, self.all_metrics[-1], self.data_dir)

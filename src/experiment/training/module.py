"""LightningModule wrapper for GPT training."""

from typing import override
import lightning as L
import torch
import torch.nn as nn

from experiment.config import TrainingConfig
from experiment.model.gpt import GPT
from experiment.training.optimizer import configure_optimizer
from experiment.training.scheduler import configure_scheduler


class GPTModule(L.LightningModule):
    """Wrap a GPT model for training with PyTorch Lightning.

    Args:
        model: The GPT model to train.
        config: Full training configuration.
        epoch_length: Number of batches per epoch; needed to configure the LR scheduler.
    """

    def __init__(self, model: GPT, config: TrainingConfig, epoch_length: int):
        super().__init__()
        self.model = model
        self.config = config
        self.epoch_length = epoch_length
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @override
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        xb, yb = batch
        logits = self(xb)
        loss = self.criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    @override
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        xb, yb = batch
        logits = self(xb)
        loss = self.criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    @override
    def configure_optimizers(self):
        optimizer = configure_optimizer(self.model, self.config.optimizer)
        scheduler = configure_scheduler(optimizer, self.config.scheduler, epoch_length=self.epoch_length)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }

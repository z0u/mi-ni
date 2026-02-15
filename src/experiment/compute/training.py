from pathlib import Path

import torch

from experiment.compute.data_pipelines import load_data
from experiment.compute.model import save_checkpoint
from experiment.config import TrainingConfig
from experiment.data.dataloader import get_dataloader
from experiment.model.gpt import GPT
from experiment.training.metrics import TrainingMetrics
from experiment.training.optimizer import configure_optimizer
from experiment.training.scheduler import configure_scheduler
from mini.progress import emit_progress
from utils.torch.mixed_precision import AMPContext
from utils.torch.types import get_device


def train_model(
    config: TrainingConfig,
    data_dir: Path,
    checkpoint_every: int | None = None,
) -> tuple[GPT, list[TrainingMetrics]]:
    """Train a model and return it with per-epoch metrics.

    Args:
        config: Full training configuration.
        data_dir: Directory for loading data and saving checkpoints.
        checkpoint_every: Save a checkpoint every N epochs. None = only at the end.
    """
    data, metadata = load_data(data_dir)
    assert metadata.tokenizer_config.vocab_size <= config.model.vocab_size, 'Vocab size mismatch'

    model = GPT(config.model)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    if torch.cuda.is_available():
        data = data.cuda()
        model = model.cuda()
        criterion = criterion.cuda()

    train_loader, val_loader = get_dataloader(data, config.data, config.model)

    optimizer = configure_optimizer(model, config.optimizer)
    scheduler = configure_scheduler(optimizer, config.scheduler, epoch_length=len(train_loader))
    amp_context = AMPContext(use_amp=config.amp.enabled, device_type=get_device(model), dtype=config.amp.dtype)

    steps_per_epoch = len(train_loader) + len(val_loader)
    total_steps = config.scheduler.epochs * steps_per_epoch
    step = 0
    all_metrics: list[TrainingMetrics] = []

    if checkpoint_every is None:
        checkpoint_every = max(1, config.scheduler.epochs // 50)

    for epoch in range(config.scheduler.epochs):
        model.train()
        for xb, yb in train_loader:
            with amp_context.forward_pass():
                logits = model(xb)
                loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            amp_context.backward_pass(loss, optimizer)
            scheduler.step()
            step += 1
            emit_progress(step, total_steps, message=f'epoch {epoch + 1}/{config.scheduler.epochs} train')

        total_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                with amp_context.forward_pass():
                    logits = model(xb)
                    loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                total_val_loss += loss.item()
                step += 1
                emit_progress(step, total_steps, message=f'epoch {epoch + 1}/{config.scheduler.epochs} val')

        metrics = TrainingMetrics(
            epoch=epoch,
            learning_rate=float(scheduler.get_last_lr()[0]),
            val_loss=total_val_loss / len(val_loader),
            training_tokens=(epoch + 1) * len(train_loader) * config.data.batch_size * config.model.block_size,
        )
        all_metrics.append(metrics)

        if epoch > 0 and epoch % checkpoint_every == 0 or epoch == config.scheduler.epochs - 1:
            save_checkpoint(model, config, metrics, data_dir)

    return model, all_metrics

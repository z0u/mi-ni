from typing import Generator, Literal

import torch

from experiment.compute.data_pipelines import load_data
from experiment.config import TrainingConfig
from experiment.data.dataloader import get_dataloader
from experiment.model.gpt import GPT
from experiment.training.metrics import TrainingMetrics
from experiment.training.optimizer import configure_optimizer
from experiment.training.scheduler import configure_scheduler
from utils.training import mode

TrainingEvent = (
    tuple[Literal['epochs', 'steps-per-epoch', 'train-step', 'val-step'], int]
    | tuple[Literal['epoch-end'], TrainingMetrics]
    | tuple[Literal['checkpoint'], tuple[GPT, TrainingConfig, TrainingMetrics]]
)


def train_model(config: TrainingConfig) -> Generator[TrainingEvent]:
    # Load data
    data, metadata = load_data()
    assert metadata.tokenizer_config.vocab_size <= config.model.vocab_size, 'Vocab size mismatch'
    train_loader, val_loader = get_dataloader(data, config.data, config.model)

    # Model
    model = GPT(config.model)
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        data = data.cuda()
        model = model.cuda()
        criterion = criterion.cuda()

    # Optimizer and scheduler
    optimizer = configure_optimizer(model, config.optimizer)
    scheduler = configure_scheduler(optimizer, config.scheduler, epoch_length=len(train_loader))

    # Training loop
    yield 'epochs', config.scheduler.epochs
    yield 'steps-per-epoch', len(train_loader) + len(val_loader)
    for i in range(config.scheduler.epochs):
        for xb, yb in train_loader:
            yield 'train-step', 1
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

        total_val_loss = 0
        with mode(model, 'eval'), torch.no_grad():
            for xb, yb in val_loader:
                yield 'val-step', 1
                logits = model(xb)
                loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                total_val_loss += loss.item()

        metrics = TrainingMetrics(
            epoch=i,
            learning_rate=scheduler.get_last_lr()[0],
            val_loss=total_val_loss / len(val_loader),
            training_tokens=(i + 1) * len(train_loader) * config.data.batch_size * config.model.block_size,
        )

        yield 'checkpoint', (model, config, metrics)
        yield 'epoch-end', metrics

import torch
from jaxtyping import Int
from torch.utils.data import DataLoader

from experiment.config import DataConfig, ModelConfig
from experiment.data.dataset import OverlappingRandomSampler, TextDataset
from utils.param_types import validate_call


@validate_call
def get_dataloader(
    data: Int[torch.Tensor, ' T'],
    data_config: DataConfig,
    model_config: ModelConfig,
) -> tuple[DataLoader, DataLoader]:
    n = int(data_config.train_split * len(data))
    train_dataset = TextDataset(data[:n], model_config.block_size, data_config.padding_chance)
    val_dataset = TextDataset(data[n:], model_config.block_size, data_config.padding_chance)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        sampler=OverlappingRandomSampler(train_dataset, data_config.batch_size, model_config.block_size, oversample=2),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        sampler=OverlappingRandomSampler(val_dataset, data_config.batch_size, model_config.block_size),
    )
    return train_loader, val_loader

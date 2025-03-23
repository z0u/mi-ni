import math
from typing import Sized

import torch
from jaxtyping import Int
from pydantic import NonNegativeFloat, NonNegativeInt
from torch.utils.data import Dataset, RandomSampler

from utils.param_types import validate_call


class TextDataset(Dataset):
    @validate_call
    def __init__(self, data: Int[torch.Tensor, ' T'], block_size: NonNegativeInt, padding_chance=0.1):
        self.data = data
        self.block_size = block_size
        self.padding_chance = padding_chance

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: NonNegativeInt) -> tuple[Int[torch.Tensor, ' T'], Int[torch.Tensor, ' T']]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]

        # Randomly pad the beginning of the sequence
        if self.padding_chance and torch.rand(1).item() < self.padding_chance:
            pad_length = torch.randint(1, self.block_size // 3, (1,)).item()
            x[:pad_length] = 0
            if pad_length > 1:
                y[: pad_length - 1] = 0

        return x, y


class OverlappingRandomSampler(RandomSampler):
    @validate_call
    def __init__(
        self,
        data_source: Sized,
        batch_size: NonNegativeInt,
        seq_len: NonNegativeInt,
        oversample: NonNegativeFloat = 1,
    ):
        num_samples = math.ceil(len(data_source) * oversample / batch_size / seq_len)
        super().__init__(data_source, replacement=True, num_samples=num_samples)

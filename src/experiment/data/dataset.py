import math
from typing import Sized

import torch
from jaxtyping import Int
from pydantic import NonNegativeFloat, NonNegativeInt
from torch.utils.data import Dataset, RandomSampler

from utils.param_types import validate_call


class TextDataset(Dataset):
    @validate_call(validate_return=True)
    def __init__(self, data: Int[torch.Tensor, " T"], block_size: NonNegativeInt):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: NonNegativeInt) -> tuple[Int[torch.Tensor, " T"], Int[torch.Tensor, " T"]]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


class OverlappingRandomSampler(RandomSampler):
    @validate_call(validate_return=True)
    def __init__(
        self,
        data_source: Sized,
        batch_size: NonNegativeInt,
        seq_len: NonNegativeInt,
        oversample: NonNegativeFloat = 1,
    ):
        num_samples = math.ceil(len(data_source) * oversample / batch_size / seq_len)
        super().__init__(data_source, replacement=True, num_samples=num_samples)

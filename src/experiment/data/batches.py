"""Random-crop batch sampling over a tokenized corpus.

Replaces the torch Dataset/DataLoader/Sampler trio with plain numpy: each batch
is a set of random substrings of the corpus (which may overlap), with the
targets shifted right by one token. Sequences are occasionally given a padded
(zeroed) prefix so the model learns to cope with short contexts.
"""

import math
from typing import Iterator

import numpy as np
from jaxtyping import Int

from experiment.config import DataConfig, ModelConfig
from utils.param_types import validate_call


@validate_call
def split_data(
    data: Int[np.ndarray, " T"],
    train_split: float,
) -> tuple[Int[np.ndarray, " Tt"], Int[np.ndarray, " Tv"]]:
    """Split the corpus into contiguous train and validation portions."""
    n = int(train_split * len(data))
    return data[:n], data[n:]


@validate_call
def batches_per_epoch(
    n_tokens: int,
    data_config: DataConfig,
    model_config: ModelConfig,
    oversample: float | None = None,
) -> int:
    """Number of batches that make up one epoch over a corpus of *n_tokens*.

    Each epoch covers roughly `oversample / batch_size` of the corpus. (Sizing
    inherited from the original sampler-based loader, so notebook results stay
    comparable across the JAX port.)
    """
    if oversample is None:
        oversample = data_config.oversample
    n_starts = max(1, n_tokens - model_config.block_size - 1)
    n_samples = math.ceil(n_starts * oversample / (data_config.batch_size * model_config.block_size))
    return max(1, math.ceil(n_samples / data_config.batch_size))


@validate_call
def sample_batches(
    data: Int[np.ndarray, " T"],
    data_config: DataConfig,
    model_config: ModelConfig,
    n_batches: int,
    rng: np.random.Generator,
) -> Iterator[tuple[Int[np.ndarray, "B T"], Int[np.ndarray, "B T"]]]:
    """Yield *n_batches* of (inputs, targets): random crops with targets shifted by one."""
    block_size = model_config.block_size
    n_starts = len(data) - block_size - 1
    if n_starts < 1:
        raise ValueError(f"Corpus of {len(data)} tokens is too short for block size {block_size}")

    for _ in range(n_batches):
        starts = rng.integers(0, n_starts, size=data_config.batch_size)
        x = np.stack([data[s : s + block_size] for s in starts])
        y = np.stack([data[s + 1 : s + block_size + 1] for s in starts])

        # Randomly pad the beginning of some sequences
        if data_config.padding_chance:
            for i in np.flatnonzero(rng.random(len(starts)) < data_config.padding_chance):
                pad_length = int(rng.integers(1, block_size // 3))
                x[i, :pad_length] = 0
                if pad_length > 1:
                    y[i, : pad_length - 1] = 0

        yield x, y

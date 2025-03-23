import torch.nn as nn

from experiment.config import ModelConfig


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # All operations maintain the batch and sequence length dimensions,
        # only transforming the embedding dimension.
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_ff),
            nn.GELU(),
            nn.Linear(config.n_ff, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

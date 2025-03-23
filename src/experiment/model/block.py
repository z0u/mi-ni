import torch.nn as nn

from experiment.config import ModelConfig
from experiment.model.attention import CausalSelfAttention
from experiment.model.mlp import MLP


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # log.info(f"Block input shape: {x.shape}")
        # x+ is the residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

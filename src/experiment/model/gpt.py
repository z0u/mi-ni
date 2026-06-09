"""Baseline GPT: pre-norm transformer with additive residuals.

The conventional architecture, and mi-ni's default. Each block layer-norms its
input, runs a sub-module, and adds the result back to the residual stream. The
`ngpt` module tells the alternative, normalized story.
"""

import logging

import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional as F

from experiment.config import ModelConfig
from experiment.model._shared import LanguageModel, RotaryEncoding, merge_heads, split_heads

log = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kq = config.n_head_dim
        self.n_v = config.n_head_dim
        self.n_kq_tot = self.n_head * self.n_kq
        self.n_v_tot = self.n_head * self.n_v

        # Projections to total attention dim (q, k, v) and back.
        self.qkv = nn.Linear(config.n_embd, 2 * self.n_kq_tot + self.n_v_tot)
        self.proj = nn.Linear(self.n_v_tot, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = self.n_kq**-0.5

    def forward(self, x: Float[Tensor, 'B T C'], enc: RotaryEncoding):
        B, T, _C = x.shape
        q, k, v = self.qkv(x).split([self.n_kq_tot, self.n_kq_tot, self.n_v_tot], dim=-1)
        q = split_heads(q, self.n_head)
        k = split_heads(k, self.n_head)
        v = split_heads(v, self.n_head)

        q, k = enc(q, k)

        # Scaled dot-product attention with causal masking.
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(att.new_ones(T, T).tril() == 0, float('-inf'))
        att = self.dropout(F.softmax(att, dim=-1))
        y = att @ v

        y = merge_heads(y)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_ff),
            nn.GELU(),
            nn.Linear(config.n_ff, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, enc: RotaryEncoding):
        x = x + self.attn(self.ln_1(x), enc)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.rotary_enc = RotaryEncoding(config.n_head_dim, block_size=config.block_size)
        self.ln_f = nn.LayerNorm(config.n_embd)


class GPT(LanguageModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        config = self.config

        log.info('Initializing GPT model with config: %s', config)
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tie weights between embedding and LM head
        self.transformer.wte.weight = self.lm_head.weight

        log.info('number of parameters: %.2fM', self.get_num_params() / 1e6)

    def forward(self, idx: Int[Tensor, 'B T']):
        x: Float[Tensor, 'B T C'] = self.transformer.wte(idx)
        enc = self.transformer.rotary_enc
        for block in self.transformer.blocks:
            x = block(x, enc)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

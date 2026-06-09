"""nGPT: every activation and weight matrix lives on the unit hypersphere.

Two flavours, selected by `config.ngpt_variant`:

- `'crude'` (default, first-class): scalar gains everywhere, and a gated additive
  retraction for the residual (`h + α·ĥ*`, then re-normalize). A handful of
  learnable numbers per layer — the minimal thing that recovers nGPT.
- `'full'` (notebook ablation): per-channel eigen learning rates and a true
  normalized LERP residual (`h + α·(ĥ* − h)`), i.e. nGPT as published.

The empirical finding is that `'crude'` matches `'full'`, so the per-channel
machinery is carried only for the ablation, not shipped as the default.
"""

import logging
from typing import cast

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional as F

from experiment.config import ModelConfig
from experiment.model._shared import LanguageModel, RotaryEncoding, Scale, merge_heads, split_heads

log = logging.getLogger(__name__)


def _is_full(config: ModelConfig) -> bool:
    return config.ngpt_variant == 'full'


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kq = config.n_head_dim
        self.n_v = config.n_head_dim
        self.n_kq_tot = self.n_head * self.n_kq
        self.n_v_tot = self.n_head * self.n_v

        # Bias-free: biases would push activations off the unit hypersphere.
        self.qkv = nn.Linear(config.n_embd, 2 * self.n_kq_tot + self.n_v_tot, bias=False)
        self.proj = nn.Linear(self.n_v_tot, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Per-head q/k are unit-normalized, so their dot product is a cosine in
        # [−1, 1] and needs sharpening back up before softmax.
        self.full = _is_full(config)
        if self.full:
            # Per-dim gain on q and k, then a fixed √d_k score scale.
            self.s_qk = Scale(self.n_kq, init=1.0, scale=config.n_embd**-0.5)
            self.qk_scale = self.n_kq**0.5
        else:
            # A single learnable scalar temperature, initialized to √d_k.
            self.s_qk = Scale(1, init=self.n_kq**0.5, scale=config.n_embd**-0.5)
            self.qk_scale = 1.0

    def forward(self, x: Float[Tensor, 'B T C'], enc: RotaryEncoding):
        B, T, _C = x.shape
        q, k, v = self.qkv(x).split([self.n_kq_tot, self.n_kq_tot, self.n_v_tot], dim=-1)
        q = split_heads(q, self.n_head)
        k = split_heads(k, self.n_head)
        v = split_heads(v, self.n_head)

        q, k = enc(q, k)

        # Normalize q and k onto the unit hypersphere (per head). RoPE is a
        # rotation, so it commutes with normalization.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        if self.full:
            q = q * self.s_qk()
            k = k * self.s_qk()

        att = (q @ k.transpose(-2, -1)) * (self.qk_scale if self.full else self.s_qk())
        att = att.masked_fill(att.new_ones(T, T).tril() == 0, float('-inf'))
        att = self.dropout(F.softmax(att, dim=-1))
        y = att @ v

        y = merge_heads(y)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Bias-free to keep activations on the unit hypersphere.
        self.fc = nn.Linear(config.n_embd, config.n_ff, bias=False)
        self.proj = nn.Linear(config.n_ff, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        # With unit-norm input and unit-norm weights the pre-activations would be
        # ~1/√d — far too small, leaving GELU near-linear. Scale the up-projection
        # by a √n_embd baseline (times learnable s_u) so GELU sees O(1) inputs.
        self.s_u = Scale(config.n_ff if _is_full(config) else 1, init=1.0, scale=1.0)
        self.su_base = config.n_embd**0.5

    def forward(self, h):
        u = self.fc(h) * (self.s_u() * self.su_base)
        return self.dropout(self.proj(F.gelu(u)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.full = _is_full(config)
        # Residual gates: 'full' uses per-channel eigen learning rates, 'crude' a
        # single scalar step size per sub-module (ReZero/LayerScale-style).
        n = config.n_embd if self.full else 1
        scale = config.n_embd**-0.5
        self.alpha_a = Scale(n, init=0.05, scale=scale)
        self.alpha_m = Scale(n, init=0.05, scale=scale)

    def forward(self, h, enc: RotaryEncoding):
        # h is on the unit hypersphere; each sub-module consumes it directly.
        if self.full:
            # Normalized LERP toward the sub-module's output: h + α(ĥ* − h).
            h_a = F.normalize(self.attn(h, enc), dim=-1)
            h = F.normalize(h + self.alpha_a() * (h_a - h), dim=-1)
            h_m = F.normalize(self.mlp(h), dim=-1)
            h = F.normalize(h + self.alpha_m() * (h_m - h), dim=-1)
        else:
            # Gated additive retraction: small step toward the output, re-project.
            h = F.normalize(h + self.alpha_a() * self.attn(h, enc), dim=-1)
            h = F.normalize(h + self.alpha_m() * self.mlp(h), dim=-1)
        return h


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.rotary_enc = RotaryEncoding(config.n_head_dim, block_size=config.block_size)


class NGPT(LanguageModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        config = self.config

        log.info('Initializing nGPT (%s) model with config: %s', config.ngpt_variant, config)
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tie weights between embedding and LM head
        self.transformer.wte.weight = self.lm_head.weight
        # Learnable scalar logit temperature (the hidden state is unit-norm, so
        # raw logits would be cosines in [−1, 1]).
        self.s_z = Scale(1, init=1.0, scale=config.n_embd**-0.5)

        # Start every matrix on the unit hypersphere.
        self.normalize_weights()

        log.info('number of parameters: %.2fM', self.get_num_params() / 1e6)

    def forward(self, idx: Int[Tensor, 'B T']):
        # Token embeddings, projected onto the unit hypersphere.
        x: Float[Tensor, 'B T C'] = F.normalize(self.transformer.wte(idx), dim=-1)

        enc = self.transformer.rotary_enc
        for block in self.transformer.blocks:
            x = block(x, enc)

        # Hidden state is already normalized, so just project and apply the
        # learnable logit temperature.
        return self.lm_head(x) * self.s_z()

    @torch.no_grad()
    def normalize_weights(self):
        """Project every hidden-dim matrix back onto the unit hypersphere.

        Call after each optimizer step to enforce nGPT's weight constraint.
        Matrices that read from the residual stream are normalized over their
        input axis (dim=1); matrices that write to it, over their output axis
        (dim=0). `lm_head` shares its tensor with `wte`, so it is covered once.
        """
        self.transformer.wte.weight.copy_(F.normalize(self.transformer.wte.weight, dim=1))
        for block in cast(list[Block], self.transformer.blocks):
            block.attn.qkv.weight.copy_(F.normalize(block.attn.qkv.weight, dim=1))
            block.attn.proj.weight.copy_(F.normalize(block.attn.proj.weight, dim=0))
            block.mlp.fc.weight.copy_(F.normalize(block.mlp.fc.weight, dim=1))
            block.mlp.proj.weight.copy_(F.normalize(block.mlp.proj.weight, dim=0))

    @torch.no_grad()
    def scale_report(self) -> dict[str, list[float] | float]:
        """Read back the learned scalar gates and temperatures, per layer.

        With the crude (scalar) variant these are all single numbers, so we can
        see exactly what training settled on — e.g. whether the residual gates α
        land near 1/n_layer. (With 'full' these are per-channel means.)
        """
        blocks = cast(list[Block], self.transformer.blocks)
        return {
            'alpha_a': [b.alpha_a().mean().item() for b in blocks],
            'alpha_m': [b.alpha_m().mean().item() for b in blocks],
            's_qk': [b.attn.s_qk().mean().item() for b in blocks],
            's_u': [b.mlp.s_u().mean().item() for b in blocks],
            's_z': self.s_z().item(),
        }

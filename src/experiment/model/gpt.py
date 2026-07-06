"""Baseline GPT: pre-norm transformer with additive residuals.

The conventional architecture, and mi-ni's default. Each block layer-norms its
input, runs a sub-module, and adds the result back to the residual stream. The
`ngpt` module tells the alternative, normalized story.
"""

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from experiment.config import ModelConfig
from experiment.model._shared import (
    LanguageModel,
    LayerNorm,
    Linear,
    RotaryEncoding,
    merge_heads,
    split_heads,
    split_keys,
)

log = logging.getLogger(__name__)


class CausalSelfAttention(eqx.Module):
    n_head: int = eqx.field(static=True)
    n_kq_tot: int = eqx.field(static=True)
    n_v_tot: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    qkv: Linear
    proj: Linear
    dropout: eqx.nn.Dropout

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        self.n_head = config.n_head
        self.n_kq_tot = config.n_head * config.n_head_dim
        self.n_v_tot = config.n_head * config.n_head_dim
        self.scale = config.n_head_dim**-0.5

        # Projections to total attention dim (q, k, v) and back.
        qkv_key, proj_key = jr.split(key)
        self.qkv = Linear(config.n_embd, 2 * self.n_kq_tot + self.n_v_tot, key=qkv_key)
        self.proj = Linear(self.n_v_tot, config.n_embd, key=proj_key)
        self.dropout = eqx.nn.Dropout(config.dropout)

    def __call__(self, x: Float[Array, "B T C"], enc: RotaryEncoding, *, key: PRNGKeyArray | None = None):
        _B, T, _C = x.shape
        q, k, v = jnp.split(self.qkv(x), [self.n_kq_tot, 2 * self.n_kq_tot], axis=-1)
        q = split_heads(q, self.n_head)
        k = split_heads(k, self.n_head)
        v = split_heads(v, self.n_head)

        q, k = enc(q, k)

        # Scaled dot-product attention with causal masking.
        att = (q @ k.swapaxes(-2, -1)) * self.scale
        att = jnp.where(jnp.tril(jnp.ones((T, T), bool)), att, -jnp.inf)
        att_key, out_key = split_keys(key, 2)
        att = self.dropout(jax.nn.softmax(att, axis=-1), key=att_key)
        y = att @ v

        y = merge_heads(y)
        return self.dropout(self.proj(y), key=out_key)


class MLP(eqx.Module):
    fc: Linear
    proj: Linear
    dropout: eqx.nn.Dropout

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        fc_key, proj_key = jr.split(key)
        self.fc = Linear(config.n_embd, config.n_ff, key=fc_key)
        self.proj = Linear(config.n_ff, config.n_embd, key=proj_key)
        self.dropout = eqx.nn.Dropout(config.dropout)

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        return self.dropout(self.proj(jax.nn.gelu(self.fc(x), approximate=False)), key=key)


class Block(eqx.Module):
    ln_1: LayerNorm
    attn: CausalSelfAttention
    ln_2: LayerNorm
    mlp: MLP

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        attn_key, mlp_key = jr.split(key)
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, key=attn_key)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config, key=mlp_key)

    def __call__(self, x, enc: RotaryEncoding, *, key: PRNGKeyArray | None = None):
        attn_key, mlp_key = split_keys(key, 2)
        x = x + self.attn(self.ln_1(x), enc, key=attn_key)
        x = x + self.mlp(self.ln_2(x), key=mlp_key)
        return x


class Transformer(eqx.Module):
    wte: Float[Array, "V C"]
    blocks: tuple[Block, ...]
    rotary_enc: RotaryEncoding
    ln_f: LayerNorm

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        wte_key, *block_keys = jr.split(key, config.n_layer + 1)
        # Initialized like the (tied) LM head it doubles as.
        lim = config.n_embd**-0.5
        self.wte = jr.uniform(wte_key, (config.vocab_size, config.n_embd), minval=-lim, maxval=lim)
        self.blocks = tuple(Block(config, key=k) for k in block_keys)
        self.rotary_enc = RotaryEncoding(config.n_head_dim)
        self.ln_f = LayerNorm(config.n_embd)


class GPT(LanguageModel):
    transformer: Transformer

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        log.info("Initializing GPT model with config: %s", config)
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.transformer = Transformer(config, key=key)

        log.info("number of parameters: %.2fM", self.get_num_params() / 1e6)

    def __call__(self, idx: Int[Array, "B T"], *, key: PRNGKeyArray | None = None):
        x: Float[Array, "B T C"] = self.transformer.wte[idx]
        # Gradient-checkpoint each block: the backward pass recomputes
        # activations instead of storing every layer's O(T²) attention maps.
        enc = self.transformer.rotary_enc
        run_block = eqx.filter_checkpoint(lambda block, h, block_key: block(h, enc, key=block_key))
        for block, block_key in zip(
            self.transformer.blocks, split_keys(key, len(self.transformer.blocks)), strict=True
        ):
            x = run_block(block, x, block_key)
        x = self.transformer.ln_f(x)
        # LM head: tied to the embedding by construction (one shared array).
        return x @ self.transformer.wte.T

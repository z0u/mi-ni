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

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from experiment.config import ModelConfig
from experiment.model._shared import (
    LanguageModel,
    Linear,
    RotaryEncoding,
    Scale,
    merge_heads,
    normalize,
    split_heads,
)

log = logging.getLogger(__name__)


def _is_full(config: ModelConfig) -> bool:
    return config.ngpt_variant == "full"


class CausalSelfAttention(eqx.Module):
    n_head: int = eqx.field(static=True)
    n_kq_tot: int = eqx.field(static=True)
    n_v_tot: int = eqx.field(static=True)
    full: bool = eqx.field(static=True)
    qk_scale: float = eqx.field(static=True)

    qkv: Linear
    proj: Linear
    s_qk: Scale

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        self.n_head = config.n_head
        self.n_kq_tot = config.n_head * config.n_head_dim
        self.n_v_tot = config.n_head * config.n_head_dim

        # Bias-free: biases would push activations off the unit hypersphere.
        qkv_key, proj_key = jr.split(key)
        self.qkv = Linear(config.n_embd, 2 * self.n_kq_tot + self.n_v_tot, key=qkv_key, use_bias=False)
        self.proj = Linear(self.n_v_tot, config.n_embd, key=proj_key, use_bias=False)

        # Per-head q/k are unit-normalized, so their dot product is a cosine in
        # [−1, 1] and needs sharpening back up before softmax.
        self.full = _is_full(config)
        if self.full:
            # Per-dim gain on q and k, then a fixed √d_k score scale.
            self.s_qk = Scale(config.n_head_dim, init=1.0, scale=config.n_embd**-0.5)
            self.qk_scale = config.n_head_dim**0.5
        else:
            # A single learnable scalar temperature, initialized to √d_k.
            self.s_qk = Scale(1, init=config.n_head_dim**0.5, scale=config.n_embd**-0.5)
            self.qk_scale = 1.0

    def __call__(self, x: Float[Array, "B T C"], enc: RotaryEncoding):
        _B, T, _C = x.shape
        q, k, v = jnp.split(self.qkv(x), [self.n_kq_tot, 2 * self.n_kq_tot], axis=-1)
        q = split_heads(q, self.n_head)
        k = split_heads(k, self.n_head)
        v = split_heads(v, self.n_head)

        q, k = enc(q, k)

        # Normalize q and k onto the unit hypersphere (per head). RoPE is a
        # rotation, so it commutes with normalization.
        q = normalize(q)
        k = normalize(k)
        if self.full:
            q = q * self.s_qk()
            k = k * self.s_qk()

        att = (q @ k.swapaxes(-2, -1)) * (self.qk_scale if self.full else self.s_qk())
        att = jnp.where(jnp.tril(jnp.ones((T, T), bool)), att, -jnp.inf)
        att = jax.nn.softmax(att, axis=-1)
        y = att @ v

        y = merge_heads(y)
        return self.proj(y)


class MLP(eqx.Module):
    su_base: float = eqx.field(static=True)

    fc: Linear
    proj: Linear
    s_u: Scale

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        # Bias-free to keep activations on the unit hypersphere.
        fc_key, proj_key = jr.split(key)
        self.fc = Linear(config.n_embd, config.n_ff, key=fc_key, use_bias=False)
        self.proj = Linear(config.n_ff, config.n_embd, key=proj_key, use_bias=False)
        # With unit-norm input and unit-norm weights the pre-activations would be
        # ~1/√d — far too small, leaving GELU near-linear. Scale the up-projection
        # by a √n_embd baseline (times learnable s_u) so GELU sees O(1) inputs.
        self.s_u = Scale(config.n_ff if _is_full(config) else 1, init=1.0, scale=1.0)
        self.su_base = config.n_embd**0.5

    def __call__(self, h):
        u = self.fc(h) * (self.s_u() * self.su_base)
        return self.proj(jax.nn.gelu(u, approximate=False))


class Block(eqx.Module):
    full: bool = eqx.field(static=True)

    attn: CausalSelfAttention
    mlp: MLP
    alpha_a: Scale
    alpha_m: Scale

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        attn_key, mlp_key = jr.split(key)
        self.attn = CausalSelfAttention(config, key=attn_key)
        self.mlp = MLP(config, key=mlp_key)
        self.full = _is_full(config)
        # Residual gates: 'full' uses per-channel eigen learning rates, 'crude' a
        # single scalar step size per sub-module (ReZero/LayerScale-style).
        n = config.n_embd if self.full else 1
        scale = config.n_embd**-0.5
        self.alpha_a = Scale(n, init=0.05, scale=scale)
        self.alpha_m = Scale(n, init=0.05, scale=scale)

    def __call__(self, h, enc: RotaryEncoding):
        # h is on the unit hypersphere; each sub-module consumes it directly.
        if self.full:
            # Normalized LERP toward the sub-module's output: h + α(ĥ* − h).
            h_a = normalize(self.attn(h, enc))
            h = normalize(h + self.alpha_a() * (h_a - h))
            h_m = normalize(self.mlp(h))
            h = normalize(h + self.alpha_m() * (h_m - h))
        else:
            # Gated additive retraction: small step toward the output, re-project.
            h = normalize(h + self.alpha_a() * self.attn(h, enc))
            h = normalize(h + self.alpha_m() * self.mlp(h))
        return h


class Transformer(eqx.Module):
    wte: Float[Array, "V C"]
    blocks: tuple[Block, ...]
    rotary_enc: RotaryEncoding

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        wte_key, *block_keys = jr.split(key, config.n_layer + 1)
        # Initialized like the (tied) LM head it doubles as; immediately
        # re-projected onto the sphere by `normalize_weights` below.
        lim = config.n_embd**-0.5
        self.wte = jr.uniform(wte_key, (config.vocab_size, config.n_embd), minval=-lim, maxval=lim)
        self.blocks = tuple(Block(config, key=k) for k in block_keys)
        self.rotary_enc = RotaryEncoding(config.n_head_dim)


class NGPT(LanguageModel):
    transformer: Transformer
    s_z: Scale

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        log.info("Initializing nGPT (%s) model with config: %s", config.ngpt_variant, config)
        # nGPT is deliberately dropout-free: the unit-hypersphere constraint is the
        # regularizer, and threading PRNG keys for dropout complicates the forward.
        if config.dropout:
            raise ValueError(f"nGPT does not support dropout; set dropout=0 (got {config.dropout})")
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.transformer = Transformer(config, key=key)
        # Learnable scalar logit temperature (the hidden state is unit-norm, so
        # raw logits would be cosines in [−1, 1]).
        self.s_z = Scale(1, init=1.0, scale=config.n_embd**-0.5)

        # Start every matrix on the unit hypersphere.
        normalized = self.normalize_weights()
        self.transformer = normalized.transformer

        log.info("number of parameters: %.2fM", self.get_num_params() / 1e6)

    def __call__(self, idx: Int[Array, "B T"], *, key: PRNGKeyArray | None = None):
        # Token embeddings, projected onto the unit hypersphere.
        x: Float[Array, "B T C"] = normalize(self.transformer.wte[idx])

        # Gradient-checkpoint each block: the backward pass recomputes
        # activations instead of storing every layer's O(T²) attention maps.
        enc = self.transformer.rotary_enc
        run_block = eqx.filter_checkpoint(lambda block, h: block(h, enc))
        for block in self.transformer.blocks:
            x = run_block(block, x)

        # Hidden state is already normalized, so just project (tied LM head) and
        # apply the learnable logit temperature.
        return (x @ self.transformer.wte.T) * self.s_z()

    def normalize_weights(self) -> "NGPT":
        """Project every hidden-dim matrix back onto the unit hypersphere.

        Apply after each optimizer step to enforce nGPT's weight constraint:
        `model = model.normalize_weights()`. Matrices that read from the
        residual stream are normalized over their input axis (axis=1); matrices
        that write to it, over their output axis (axis=0). The LM head shares
        the embedding array, so it is covered once.
        """

        def where(m: NGPT):
            return (
                [m.transformer.wte]
                + [b.attn.qkv.weight for b in m.transformer.blocks]
                + [b.attn.proj.weight for b in m.transformer.blocks]
                + [b.mlp.fc.weight for b in m.transformer.blocks]
                + [b.mlp.proj.weight for b in m.transformer.blocks]
            )  # fmt: skip

        def replacements(m: NGPT):
            return (
                [normalize(m.transformer.wte, axis=1)]
                + [normalize(b.attn.qkv.weight, axis=1) for b in m.transformer.blocks]
                + [normalize(b.attn.proj.weight, axis=0) for b in m.transformer.blocks]
                + [normalize(b.mlp.fc.weight, axis=1) for b in m.transformer.blocks]
                + [normalize(b.mlp.proj.weight, axis=0) for b in m.transformer.blocks]
            )  # fmt: skip

        return eqx.tree_at(where, self, replacements(self))

    def scale_report(self) -> dict[str, list[float] | float]:
        """Read back the learned scalar gates and temperatures, per layer.

        With the crude (scalar) variant these are all single numbers, so we can
        see exactly what training settled on — e.g. whether the residual gates α
        land near 1/n_layer. (With 'full' these are per-channel means.)
        """
        blocks = self.transformer.blocks
        return {
            "alpha_a": [float(jnp.mean(b.alpha_a())) for b in blocks],
            "alpha_m": [float(jnp.mean(b.alpha_m())) for b in blocks],
            "s_qk": [float(jnp.mean(b.attn.s_qk())) for b in blocks],
            "s_u": [float(jnp.mean(b.mlp.s_u())) for b in blocks],
            "s_z": float(jnp.mean(self.s_z())),
        }

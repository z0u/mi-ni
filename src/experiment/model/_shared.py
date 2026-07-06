"""Primitives shared by every model variant.

The `gpt` (baseline) and `ngpt` (normalized) modules each tell one architecture's
story end to end; everything that does *not* vary between them lives here:
last-axis `Linear`/`LayerNorm` layers, positional rotary encoding, the learnable
`Scale`, head reshaping, the sampling loop, and the `Generation` containers it
produces.

Models here are Equinox modules: pytrees of arrays transformed by JAX. Forward
passes are pure functions — randomness (dropout, sampling) enters only through
explicit PRNG keys, and "mutating" weights means building a new model.
"""

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray
from pydantic import BaseModel, NonNegativeFloat, PositiveInt, model_validator

log = logging.getLogger(__name__)


def normalize(x: Array, axis: int = -1, eps: float = 1e-12) -> Array:
    """Project onto the unit hypersphere along *axis* (like `F.normalize`)."""
    return x / jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=True), eps)


def split_keys(key: PRNGKeyArray | None, n: int) -> tuple[PRNGKeyArray | None, ...]:
    """Split a PRNG key n ways, or pass `None` through (inference: no dropout)."""
    return (None,) * n if key is None else tuple(jr.split(key, n))


class Linear(eqx.Module):
    """Affine map over the last axis of an input of any rank.

    Unlike `eqx.nn.Linear` (which expects a single unbatched vector), this
    broadcasts over leading axes, so modules can stay written in the batched
    (B, T, C) style.
    """

    weight: Float[Array, "out in"]
    bias: Float[Array, " out"] | None

    def __init__(self, in_features: int, out_features: int, *, key: PRNGKeyArray, use_bias: bool = True):
        wkey, bkey = jr.split(key)
        lim = in_features**-0.5
        self.weight = jr.uniform(wkey, (out_features, in_features), minval=-lim, maxval=lim)
        self.bias = jr.uniform(bkey, (out_features,), minval=-lim, maxval=lim) if use_bias else None

    def __call__(self, x: Float[Array, "... in"]) -> Float[Array, "... out"]:
        y = x @ self.weight.T
        return y if self.bias is None else y + self.bias


class LayerNorm(eqx.Module):
    """Layer normalization over the last axis, broadcasting over the rest."""

    weight: Float[Array, " dim"]
    bias: Float[Array, " dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = jnp.ones(dim)
        self.bias = jnp.zeros(dim)
        self.eps = eps

    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(var + self.eps) * self.weight + self.bias


class RotaryEncoding(eqx.Module):
    """Rotary positional encoding (RoPE), applied per head during attention.

    Passed into the forward pass rather than owned by the attention module, so a
    single instance can be shared across all layers. Holds no arrays — the
    sin/cos tables are derived from the sequence length at call time (and
    constant-folded under jit), so there are no buffers for the optimizer to
    mistake for parameters.
    """

    n_head_dim: int = eqx.field(static=True)
    base: float = eqx.field(static=True)

    def __init__(self, n_head_dim: int, base: float = 10_000):
        self.n_head_dim = n_head_dim
        self.base = base

    def __call__(self, q: Float[Array, "B H T D"], k: Float[Array, "B H T D"]):
        T = q.shape[-2]
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.n_head_dim, 2) / self.n_head_dim))
        enc = jnp.concatenate((f := jnp.outer(jnp.arange(T), inv_freq), f), axis=-1)  # (T, n_head_dim)
        sin, cos = jnp.sin(enc), jnp.cos(enc)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    def _rotate_half(self, x: Array):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)


class Scale(eqx.Module):
    """Learnable scalar (n=1) or per-channel (n=d) gain with nGPT's reparametrization.

    Store the parameter at `scale`; the effective value is `param * (init / scale)`,
    so Adam's step dynamics are decoupled from the value's magnitude.
    """

    weight: Float[Array, " n"]
    forward_scale: float = eqx.field(static=True)

    def __init__(self, n: int, init: float, scale: float):
        self.forward_scale = init / scale
        self.weight = jnp.full((n,), scale)

    def __call__(self) -> Array:
        return self.weight * self.forward_scale


def split_heads(x: Float[Array, "B T C"], n_head: int) -> Float[Array, "B H T D"]:
    """Reshape (B, T, n_head * n_head_dim) into (B, n_head, T, n_head_dim)."""
    B, T, C = x.shape
    return x.reshape(B, T, n_head, C // n_head).swapaxes(1, 2)


def merge_heads(x: Float[Array, "B H T D"]) -> Float[Array, "B T C"]:
    """Reshape (B, n_head, T, n_head_dim) back into (B, T, n_head * n_head_dim)."""
    B, n_head, T, d = x.shape
    return x.swapaxes(1, 2).reshape(B, T, n_head * d)


class LanguageModel(eqx.Module):
    """Base for the model variants: holds the key dimensions and the sampling machinery.

    Subclasses build `self.transformer` and implement `__call__` mapping token
    indices (B, T) to logits (B, T, vocab). `normalize_weights` returns the model
    unchanged here and is overridden by variants that enforce the
    unit-hypersphere weight constraint.
    """

    block_size: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)

    def __call__(self, idx: Int[Array, "B T"], *, key: PRNGKeyArray | None = None) -> Float[Array, "B T V"]:
        raise NotImplementedError

    def normalize_weights(self) -> "LanguageModel":
        """Re-project weights onto the unit hypersphere; identity unless overridden."""
        return self

    def get_num_params(self) -> int:
        """Calculate the number of parameters in the model."""
        return sum(x.size for x in jax.tree.leaves(eqx.filter(self, eqx.is_array)))

    def generate(
        self,
        tok_idx: Int[Array, "B T"] | Int[np.ndarray, "B T"],
        max_new_tokens: PositiveInt,
        temperature: NonNegativeFloat = 1.0,
        pad_token_id: int = 0,
        *,
        key: PRNGKeyArray,
    ) -> "Generation":
        model = eqx.nn.inference_mode(self)
        forward = eqx.filter_jit(model.__call__)

        # Align all metric arrays to input length + max_new_tokens.
        seq = np.asarray(tok_idx)
        B, T = seq.shape
        entropies = np.full((B, T + max_new_tokens), np.nan)
        surprisals = np.full((B, T + max_new_tokens), np.nan)

        # Padding mask (True for real tokens, False for padding)
        padding_mask = seq != pad_token_id

        # Calculate metrics for the prompt (except first token)
        if T > 1:
            logits = forward(jnp.asarray(seq))
            log_probs = jax.nn.log_softmax(logits[:, :-1], axis=-1)
            probs = jnp.exp(log_probs)
            prompt_entropy = np.asarray(-jnp.sum(probs * log_probs, axis=-1))

            targets = jnp.asarray(seq[:, 1:])
            losses = np.asarray(-jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0])

            # Only store metrics for non-padding tokens
            prompt_mask = padding_mask[:, 1:T]
            surprisals[:, 1:T][prompt_mask] = losses[prompt_mask]
            entropies[:, 1:T][prompt_mask] = prompt_entropy[prompt_mask]

        # Generate tokens and track metrics
        curr_len = T
        for _ in range(max_new_tokens):
            # Feed a fixed (B, block_size) window so the jitted forward compiles
            # once. Right-pad the (≤ block_size) context to full width: causal
            # masking makes the trailing pad positions inert, and we read logits
            # at the last real position. Without padding, the growing context
            # length would retrigger a recompile every step until it fills the block.
            window = seq[:, -self.block_size :]
            context_len = window.shape[1]
            if context_len < self.block_size:
                pad = np.full((B, self.block_size - context_len), pad_token_id, dtype=window.dtype)
                window = np.concatenate([window, pad], axis=1)
            logits = forward(jnp.asarray(window))
            next_token_logits = logits[:, context_len - 1]

            # Compute raw metrics (before temperature scaling)
            log_probs = jax.nn.log_softmax(next_token_logits, axis=-1)
            probs = jnp.exp(log_probs)
            entropies[:, curr_len] = np.asarray(-jnp.sum(probs * log_probs, axis=-1))

            # Sample next token (temperature applies to sampling only)
            key, sample_key = jr.split(key)
            idx_next = jr.categorical(sample_key, next_token_logits / temperature, axis=-1)

            # Surprisal of the generated token (using raw logits for consistency)
            surprisals[:, curr_len] = np.asarray(-jnp.take_along_axis(log_probs, idx_next[:, None], axis=-1)[:, 0])

            # Append to sequence and increment position
            seq = np.concatenate([seq, np.asarray(idx_next)[:, None]], axis=1)
            curr_len += 1

        # Calculate surprise-surprise metric; NaNs propagate through.
        surprise_surprise = (surprisals - entropies) / np.log(self.vocab_size)

        return Generation(
            tokens=seq,
            vocab_size=self.vocab_size,
            surprisal=surprisals,
            entropy=entropies,
            surprise_surprise=surprise_surprise,
        )


class Generation(BaseModel, arbitrary_types_allowed=True):
    tokens: Int[np.ndarray, "B T"]
    """Generated token indices"""

    vocab_size: PositiveInt
    """Vocabulary size"""

    surprisal: Float[np.ndarray, "B T"]
    """Perplexity of each token in the sequence"""

    entropy: Float[np.ndarray, "B T"]
    """Entropy of each token in the sequence"""

    surprise_surprise: Float[np.ndarray, "B T"]
    """The normalized differences between surprisal and entropy (s2)"""

    @model_validator(mode="after")
    def same_lengths(self):
        if not (len(self.tokens) == len(self.surprisal) == len(self.entropy) == len(self.surprise_surprise)):
            raise ValueError("All tensors must be of equal length")
        return self

    def __getitem__(self, item: int):
        """Allows indexing into the Generation object"""
        return SingleGeneration(
            tokens=self.tokens[item],
            vocab_size=self.vocab_size,
            surprisal=self.surprisal[item],
            entropy=self.entropy[item],
            surprise_surprise=self.surprise_surprise[item],
        )


class SingleGeneration(BaseModel, arbitrary_types_allowed=True):
    tokens: Int[np.ndarray, " T"]
    """Generated token indices"""

    vocab_size: PositiveInt
    """Vocabulary size"""

    surprisal: Float[np.ndarray, " T"]
    """Perplexity of each token in the sequence"""

    entropy: Float[np.ndarray, " T"]
    """Entropy of each token in the sequence"""

    surprise_surprise: Float[np.ndarray, " T"]
    """The normalized differences between surprisal and entropy (s2)"""

    @model_validator(mode="after")
    def same_lengths(self):
        if not (len(self.tokens) == len(self.surprisal) == len(self.entropy) == len(self.surprise_surprise)):
            raise ValueError("All tensors must be of equal length")
        return self

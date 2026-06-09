"""Primitives shared by every model variant.

The `gpt` (baseline) and `ngpt` (normalized) modules each tell one architecture's
story end to end; everything that does *not* vary between them lives here:
positional rotary encoding, the learnable `Scale`, head reshaping, the sampling
loop, and the `Generation` containers it produces.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from pydantic import BaseModel, NonNegativeFloat, PositiveInt, model_validator
from torch import Tensor
from torch.nn import functional as F

from experiment.config import ModelConfig

log = logging.getLogger(__name__)


class RotaryEncoding(nn.Module):
    """Rotary positional encoding (RoPE), applied per head during attention.

    Passed into the forward pass rather than owned by the attention module, so a
    single instance can be shared across all layers.
    """

    sin: Tensor
    cos: Tensor

    def __init__(self, n_head_dim: int, block_size: int, base: float = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, n_head_dim, 2).float() / n_head_dim))
        positions = torch.arange(block_size)
        enc = torch.cat((f := positions.outer(inv_freq), f), dim=-1)  # (block_size, n_head_dim)
        self.register_buffer('sin', torch.sin(enc)[None, None])  # (1, 1, block_size, n_head_dim)
        self.register_buffer('cos', torch.cos(enc)[None, None])

    def forward(self, q: Tensor, k: Tensor):
        T = q.shape[-2]
        sin, cos = self.sin[:, :, :T], self.cos[:, :, :T]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    def _rotate_half(self, x: Tensor):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class Scale(nn.Module):
    """Learnable scalar (n=1) or per-channel (n=d) gain with nGPT's reparametrization.

    Store the parameter at `scale`; the effective value is `param * (init / scale)`,
    so Adam's step dynamics are decoupled from the value's magnitude.
    """

    def __init__(self, n: int, init: float, scale: float):
        super().__init__()
        self.forward_scale = init / scale
        self.weight = nn.Parameter(torch.full((n,), scale))

    def forward(self) -> Tensor:
        return self.weight * self.forward_scale


def split_heads(x: Float[Tensor, 'B T C'], n_head: int) -> Float[Tensor, 'B H T D']:
    """Reshape (B, T, n_head * n_head_dim) into (B, n_head, T, n_head_dim)."""
    B, T, C = x.shape
    return x.view(B, T, n_head, C // n_head).transpose(1, 2)


def merge_heads(x: Float[Tensor, 'B H T D']) -> Float[Tensor, 'B T C']:
    """Reshape (B, n_head, T, n_head_dim) back into (B, T, n_head * n_head_dim)."""
    B, n_head, T, d = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, n_head * d)


class LanguageModel(nn.Module):
    """Base for the model variants: holds the config and the sampling machinery.

    Subclasses build `self.transformer`/`self.lm_head` and implement `forward`.
    `normalize_weights` is a no-op here and overridden by variants that enforce
    the unit-hypersphere weight constraint.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config = config.model_copy()
        self.block_size = config.block_size

    @torch.no_grad()
    def normalize_weights(self):
        """Re-project weights onto the unit hypersphere; no-op unless overridden."""

    def get_num_params(self) -> int:
        """Calculate the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        tok_idx: Int[Tensor, 'B T'],
        max_new_tokens: PositiveInt,
        temperature: NonNegativeFloat = 1.0,
        pad_token_id: int = 0,
    ) -> 'Generation':
        # Align all metric arrays to input length + max_new_tokens.
        B, T = tok_idx.shape
        device = tok_idx.device
        entropies = torch.full((B, T + max_new_tokens), float('nan'), device=device)
        surprisals = torch.full((B, T + max_new_tokens), float('nan'), device=device)

        # Create padding mask (1 for real tokens, 0 for padding)
        padding_mask = (tok_idx != pad_token_id).bool()

        # Calculate metrics for the prompt (except first token)
        if tok_idx.size(1) > 1:
            logits = self.forward(tok_idx)
            targets = tok_idx[:, 1:]  # shifted right

            # Compute metrics without temperature scaling
            raw_probs = F.softmax(logits[:, :-1], dim=-1)
            prompt_entropy = -torch.sum(raw_probs * torch.log(raw_probs + 1e-10), dim=-1)

            losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='none',
                ignore_index=pad_token_id,
            ).view(B, -1)

            # Only store metrics for non-padding tokens
            prompt_mask = padding_mask[:, 1:T]
            surprisals[:, 1:T][prompt_mask] = losses[prompt_mask]
            entropies[:, 1:T][prompt_mask] = prompt_entropy[prompt_mask]

        # Generate tokens and track metrics
        curr_len = tok_idx.size(1)
        for _ in range(max_new_tokens):
            # Get context within block size
            inputs = tok_idx[:, -self.block_size :]
            logits = self.forward(inputs)

            # Get next token logits
            next_token_logits = logits[:, -1]

            # Compute raw metrics (before temperature scaling)
            raw_probs = F.softmax(next_token_logits, dim=-1)
            batch_entropy = -torch.sum(raw_probs * torch.log(raw_probs + 1e-10), dim=-1)
            entropies[:, curr_len] = batch_entropy

            # Apply temperature for sampling
            sampling_probs = F.softmax(next_token_logits / temperature, dim=-1)

            # Sample next token
            idx_next = torch.multinomial(sampling_probs, num_samples=1)

            # Calculate surprisal for the generated token (using raw logits for consistency)
            if curr_len > 0:
                token_loss = F.cross_entropy(next_token_logits, idx_next.view(-1), reduction='none')
                surprisals[:, curr_len] = token_loss

            # Append to sequence and increment position
            tok_idx = torch.cat([tok_idx, idx_next], dim=1)
            curr_len += 1

        # Calculate surprise-surprise metric; NaNs propagate through.
        surprise_surprise = (surprisals - entropies) / torch.log(torch.tensor(self.config.vocab_size, device=device))

        return Generation(
            tokens=tok_idx.numpy(force=True),
            vocab_size=self.config.vocab_size,
            surprisal=surprisals.numpy(force=True),
            entropy=entropies.numpy(force=True),
            surprise_surprise=surprise_surprise.numpy(force=True),
        )


class Generation(BaseModel, arbitrary_types_allowed=True):
    tokens: Int[np.ndarray, 'B T']
    """Generated token indices"""

    vocab_size: PositiveInt
    """Vocabulary size"""

    surprisal: Float[np.ndarray, 'B T']
    """Perplexity of each token in the sequence"""

    entropy: Float[np.ndarray, 'B T']
    """Entropy of each token in the sequence"""

    surprise_surprise: Float[np.ndarray, 'B T']
    """The normalized differences between surprisal and entropy (s2)"""

    @model_validator(mode='after')
    def same_lengths(self):
        if not (len(self.tokens) == len(self.surprisal) == len(self.entropy) == len(self.surprise_surprise)):
            raise ValueError('All tensors must be of equal length')
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
    tokens: Int[np.ndarray, ' T']
    """Generated token indices"""

    vocab_size: PositiveInt
    """Vocabulary size"""

    surprisal: Float[np.ndarray, ' T']
    """Perplexity of each token in the sequence"""

    entropy: Float[np.ndarray, ' T']
    """Entropy of each token in the sequence"""

    surprise_surprise: Float[np.ndarray, ' T']
    """The normalized differences between surprisal and entropy (s2)"""

    @model_validator(mode='after')
    def same_lengths(self):
        if not (len(self.tokens) == len(self.surprisal) == len(self.entropy) == len(self.surprise_surprise)):
            raise ValueError('All tensors must be of equal length')
        return self

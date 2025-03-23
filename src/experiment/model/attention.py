import torch
import torch.nn as nn
from torch.nn import functional as F
from jaxtyping import Float

from experiment.config import ModelConfig


class RotaryEncoding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, n_head_dim: int, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, n_head_dim, 2).float() / n_head_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, T: int, q: torch.Tensor, k: torch.Tensor):
        positions = torch.arange(T, device=self.inv_freq.device)
        freqs = positions.outer(self.inv_freq)
        enc = torch.cat((freqs, freqs), dim=-1)
        sin = torch.sin(enc)[None, None, :, :]
        cos = torch.cos(enc)[None, None, :, :]
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def _rotate_half(self, x: torch.Tensor):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # attention dimensions:
        # - key and query dimensions need to be the same
        # - value dimension can be different, but in practice, we set it to the same thing
        self.n_kq = config.n_head_dim
        self.n_v = config.n_head_dim
        self.n_head = config.n_head
        self.n_kq_tot = self.n_head * self.n_kq
        self.n_v_tot = self.n_head * self.n_v

        # projections from embedding dim to total attention dim (split into q, k, v) and back
        self.qkv = nn.Linear(config.n_embd, 2 * self.n_kq_tot + self.n_v_tot)
        self.proj = nn.Linear(self.n_v_tot, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.rotary_enc = RotaryEncoding(self.n_kq)

        self.n_head = config.n_head
        self.n_head_dim = config.n_head_dim
        self.scale = self.n_kq**-0.5

    def _split_heads(self, tensor: torch.Tensor, batch_size: int, seq_len: int, n_head: int, n_head_dim: int):
        """Reshape tensor from (B, T, n_head * n_head_dim) into (B, n_head, T, n_head_dim)."""
        return tensor.view(batch_size, seq_len, n_head, n_head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor, batch_size: int, seq_len: int, n_head: int, n_head_dim: int):
        """Reshape tensor from (B, n_head, T, n_head_dim) into (B, T, n_head * n_head_dim)."""
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, n_head * n_head_dim)

    def forward(self, x: Float[torch.Tensor, 'B T C']):
        # log.info(f"CausalSelfAttention input shape: {x.shape}")
        B, T, _C = x.shape

        # project all qkv matrices at once, then split
        qkv = self.qkv(x)
        q, k, v = qkv.split([self.n_kq_tot, self.n_kq_tot, self.n_v_tot], dim=-1)

        # reshape each of q, k, v from (B, T, n_head * n_head_dim) into (B, n_head, T, n_head_dim)
        q = self._split_heads(q, B, T, self.n_head, self.n_kq)
        k = self._split_heads(k, B, T, self.n_head, self.n_kq)
        v = self._split_heads(v, B, T, self.n_head, self.n_v)

        # Apply rotary encoding
        q, k = self.rotary_enc(T, q, k)

        # compute attention scores
        # 1. combine q and k to make a square matrix for query-key matching (relevance LUT)
        # 2. scale to 1/√n_kq, because otherwise the dot products grow with the number of dimensions
        # 3. causal masking
        # 4. scale relevance scores to add up to 1, so we can effectively add them up
        # 5. get the weighted contributions of all value embeddings according to their relevance
        att = q @ k.transpose(-2, -1)
        att = att * self.scale
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=att.device)) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v

        # reshape output from (B, n_head, T, n_head_dim) into (B, T, n_head * n_head_dim)
        y = self._merge_heads(y, B, T, self.n_head, self.n_v)
        # project from attention embeddings back to token embeddings
        y = self.proj(y)
        return self.dropout(y)

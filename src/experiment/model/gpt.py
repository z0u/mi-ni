import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from pydantic import BaseModel, NonNegativeFloat, PositiveInt, model_validator
from torch import Tensor
from torch.nn import functional as F

from experiment.config import ModelConfig
from experiment.model.block import Block


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config = config.model_copy()
        self.block_size = config.block_size

        print("Initializing GPT model with config:", config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                # "wpe": nn.Embedding(config.block_size, config.n_embd),  # max sequence length
                "blocks": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tie weights between embedding and LM head
        self.transformer.wte.weight = self.lm_head.weight

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, idx: Int[Tensor, "B T"]):
        # print("Forward pass with input shape:", idx.shape)
        # B, T, C = idx.shape

        # token + position embeddings
        x: Float[Tensor, "B T C"] = self.transformer.wte(idx)
        # pos = torch.arange(T, device=idx.device)
        # x += self.transformer.wpe(pos)

        # transformer blocks
        for block in self.transformer.blocks:
            x = block(x)

        # decoder head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.

        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(self, tok_idx: Int[Tensor, "B T"], max_new_tokens: PositiveInt, temperature: NonNegativeFloat = 1.0) -> "Generation":
        # Initialize metrics arrays
        # We'll align everything to input length + max_new_tokens
        B, T = tok_idx.shape
        device = tok_idx.device
        entropies = torch.full((B, T + max_new_tokens), float("nan"), device=device)
        surprisals = torch.full((B, T + max_new_tokens), float("nan"), device=device)

        # Calculate metrics for the prompt (except first token)
        if tok_idx.size(1) > 1:
            logits = self.forward(tok_idx)
            targets = tok_idx[:, 1:]  # shifted right

            # Compute metrics without temperature scaling
            raw_probs = F.softmax(logits[:, :-1], dim=-1)
            prompt_entropy = -torch.sum(raw_probs * torch.log(raw_probs + 1e-10), dim=-1)

            losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
            ).view(B, -1)

            surprisals[:, 1 : tok_idx.size(1)] = torch.exp(losses)
            entropies[:, 1 : tok_idx.size(1)] = prompt_entropy

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
                token_loss = F.cross_entropy(next_token_logits, idx_next.view(-1), reduction="none")
                surprisals[:, curr_len] = torch.exp(token_loss)

            # Append to sequence and increment position
            tok_idx = torch.cat([tok_idx, idx_next], dim=1)
            curr_len += 1

        # Calculate surprise-surprise metric
        surprise_surprise = (surprisals - entropies) / torch.log(torch.tensor(self.config.vocab_size, device=device))

        return Generation(
            tokens=tok_idx.numpy(force=True),
            surprisal=surprisals.numpy(force=True),
            entropy=entropies.numpy(force=True),
            surprise_surprise=surprise_surprise.numpy(force=True),
        )


class Generation(BaseModel, arbitrary_types_allowed=True):
    tokens: Int[np.ndarray, "B T"]
    """Generated token indices"""

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

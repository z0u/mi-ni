"""Model variants: every one runs, and nGPT keeps activations/weights on the sphere."""

import pytest
import torch
from torch.nn import functional as F

from experiment.config import ModelConfig
from experiment.model import build_model

ARCHS = [
    {'architecture': 'gpt'},
    {'architecture': 'ngpt', 'ngpt_variant': 'crude'},
    {'architecture': 'ngpt', 'ngpt_variant': 'full'},
]
NGPT_ARCHS = [a for a in ARCHS if a['architecture'] == 'ngpt']


def make_config(**overrides) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        block_size=64,
        n_embd=64,
        n_head=8,
        n_head_dim=8,
        n_ff=64,
        n_layer=2,
        dropout=0,
        **overrides,
    )


@pytest.mark.parametrize('arch', ARCHS)
def test_forward_shape_and_finite(arch):
    config = make_config(**arch)
    model = build_model(config).eval()
    idx = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(idx)
    assert logits.shape == (2, 16, config.vocab_size)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize('arch', NGPT_ARCHS)
def test_hidden_state_stays_on_sphere(arch):
    """Every nGPT block must return unit-norm hidden states."""
    config = make_config(**arch)
    model = build_model(config).eval()
    idx = torch.randint(0, config.vocab_size, (2, 16))
    enc = model.transformer.rotary_enc
    h = F.normalize(model.transformer.wte(idx), dim=-1)
    for block in model.transformer.blocks:
        h = block(h, enc)
        norms = h.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=0, atol=1e-5)


@pytest.mark.parametrize('arch', NGPT_ARCHS)
def test_normalize_weights_projects_onto_sphere(arch):
    """After normalization, each matrix is a stack of unit vectors along its hidden axis."""
    model = build_model(make_config(**arch)).eval()
    # Perturb, then re-project.
    for p in model.parameters():
        p.data.add_(torch.randn_like(p))
    model.normalize_weights()

    def unit(t: torch.Tensor, dim: int):
        norms = t.norm(dim=dim)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=0, atol=1e-5)

    unit(model.transformer.wte.weight, dim=1)
    for block in model.transformer.blocks:
        unit(block.attn.qkv.weight, dim=1)
        unit(block.attn.proj.weight, dim=0)
        unit(block.mlp.fc.weight, dim=1)
        unit(block.mlp.proj.weight, dim=0)


def test_baseline_normalize_weights_is_noop():
    """The baseline carries no hypersphere constraint, so the training hook does nothing."""
    model = build_model(make_config(architecture='gpt'))
    before = [p.clone() for p in model.parameters()]
    model.normalize_weights()
    for a, b in zip(before, model.parameters(), strict=True):
        torch.testing.assert_close(a, b)

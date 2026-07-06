"""Model variants: every one runs, and nGPT keeps activations/weights on the sphere."""

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from experiment.config import ModelConfig
from experiment.model import NGPT, build_model
from experiment.model._shared import normalize

ARCHS = [
    {"architecture": "gpt"},
    {"architecture": "ngpt", "ngpt_variant": "crude"},
    {"architecture": "ngpt", "ngpt_variant": "full"},
]
NGPT_ARCHS = [a for a in ARCHS if a["architecture"] == "ngpt"]


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


@pytest.mark.parametrize("arch", ARCHS)
def test_forward_shape_and_finite(arch):
    config = make_config(**arch)
    model = build_model(config, key=jr.key(0))
    idx = jr.randint(jr.key(1), (2, 16), 0, config.vocab_size)
    logits = model(idx)
    assert logits.shape == (2, 16, config.vocab_size)
    assert jnp.isfinite(logits).all()


@pytest.mark.parametrize("arch", NGPT_ARCHS)
def test_hidden_state_stays_on_sphere(arch):
    """Every nGPT block must return unit-norm hidden states."""
    config = make_config(**arch)
    # These tests are nGPT-only; cast past the `build_model` base return type so
    # the type checker sees the concrete attributes (`build_model` -> LanguageModel).
    model = cast(NGPT, build_model(config, key=jr.key(0)))
    idx = jr.randint(jr.key(1), (2, 16), 0, config.vocab_size)
    enc = model.transformer.rotary_enc
    h = normalize(model.transformer.wte[idx])
    for block in model.transformer.blocks:
        h = block(h, enc)
        norms = jnp.linalg.norm(h, axis=-1)
        np.testing.assert_allclose(norms, jnp.ones_like(norms), rtol=0, atol=1e-5)


@pytest.mark.parametrize("arch", NGPT_ARCHS)
def test_normalize_weights_projects_onto_sphere(arch):
    """After normalization, each matrix is a stack of unit vectors along its hidden axis."""
    model = cast(NGPT, build_model(make_config(**arch), key=jr.key(0)))
    # Perturb, then re-project.
    rng = np.random.default_rng(0)
    model = jax.tree.map(
        lambda p: p + rng.normal(size=p.shape) if eqx.is_inexact_array(p) else p,
        model,
    )
    model = model.normalize_weights()

    def unit(t, axis: int):
        norms = jnp.linalg.norm(t, axis=axis)
        np.testing.assert_allclose(norms, jnp.ones_like(norms), rtol=0, atol=1e-5)

    unit(model.transformer.wte, axis=1)
    for block in model.transformer.blocks:
        unit(block.attn.qkv.weight, axis=1)
        unit(block.attn.proj.weight, axis=0)
        unit(block.mlp.fc.weight, axis=1)
        unit(block.mlp.proj.weight, axis=0)


def test_baseline_normalize_weights_is_noop():
    """The baseline carries no hypersphere constraint, so the training hook does nothing."""
    model = build_model(make_config(architecture="gpt"), key=jr.key(0))
    normalized = model.normalize_weights()
    for a, b in zip(
        jax.tree.leaves(eqx.filter(model, eqx.is_array)),
        jax.tree.leaves(eqx.filter(normalized, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)

import equinox as eqx
import jax
import optax

from experiment.config import OptimizerConfig


def configure_optimizer(
    model: eqx.Module,
    config: OptimizerConfig,
    learning_rate: float | optax.Schedule,
) -> optax.GradientTransformation:
    # apply weight decay to weight matrices (2+D arrays) but not to biases/gains (1-D or 0-D)
    params = eqx.filter(model, eqx.is_inexact_array)
    decay_mask = jax.tree.map(lambda p: p.ndim >= 2, params)

    return optax.adamw(
        learning_rate,
        b1=config.betas[0],
        b2=config.betas[1],
        weight_decay=config.weight_decay,
        mask=decay_mask,
    )

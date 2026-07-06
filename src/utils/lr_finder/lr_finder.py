from typing import Callable, Iterator, TypeAlias, cast

import equinox as eqx
import jax.random as jr
import numpy as np
import optax
from jaxtyping import PRNGKeyArray

from mini.progress import emit_progress
from utils.lr_finder.types import LRFinderConfig, LRFinderSeries, SearchMethod
from utils.param_types import validate_call

Range: TypeAlias = tuple[float, float]
Batch: TypeAlias = tuple[np.ndarray, np.ndarray]
LossFn: TypeAlias = Callable


@validate_call
def lr_finder_search(
    model: eqx.Module,
    loss_fn: LossFn,
    make_optimizer: Callable[..., optax.GradientTransformation],
    batches: Iterator[Batch],
    start_lr: float = 1e-10,
    end_lr: float = 1e1,
    num_zooms: int = 5,
    steps_per_zoom: int = 10,
    zoom_factor: float = 0.5,
    method: SearchMethod = "steepest",
    *,
    key: PRNGKeyArray,
) -> tuple[float, LRFinderConfig, list[LRFinderSeries]]:
    """Perform multi-scale learning rate range test.

    Return the suggested learning rate, the search config, and per-zoom
    series data (for visualization).

    Args:
        model: The (initial) model to probe; never mutated — each zoom restarts
            from this model and a fresh optimizer state.
        loss_fn: `(model, inputs, targets, key) -> scalar` loss.
        make_optimizer: Builds the optimizer for a given `learning_rate=...`;
            wrapped with `optax.inject_hyperparams` so the LR can vary per step.
        batches: An (endless) iterator of `(inputs, targets)` batches.
        start_lr: Lower bound of the initial search range.
        end_lr: Upper bound of the initial search range.
        num_zooms: How many times to narrow the search range.
        steps_per_zoom: Optimization steps per zoom level.
        zoom_factor: How much of the range to keep on each zoom (log-space).
        method: How to propose the next range from the loss curve.
        key: PRNG key for dropout.
    """
    if start_lr >= end_lr:
        raise ValueError("start_lr must be less than end_lr")

    optimizer = optax.inject_hyperparams(make_optimizer)(learning_rate=start_lr)
    initial_params = eqx.filter(model, eqx.is_inexact_array)

    @eqx.filter_jit
    def test_lr(model: eqx.Module, opt_state, inputs, targets, key: PRNGKeyArray):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs, targets, key)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    total_steps = num_zooms * steps_per_zoom
    config = LRFinderConfig(
        num_zooms=num_zooms,
        method=method,
        zoom_factor=zoom_factor,
        steps_per_zoom=steps_per_zoom,
        start_lr=start_lr,
        end_lr=end_lr,
    )

    best_lr = float("nan")
    steepest_lr = float("nan")
    lowest_lr = float("inf")
    history: list[LRFinderSeries] = []

    current_range = (start_lr, end_lr)
    for zoom in range(num_zooms):
        emit_progress(zoom * steps_per_zoom, total_steps, message=f"zoom {zoom + 1}/{num_zooms}")

        lr_schedule = _get_lr_schedule(current_range, steps_per_zoom)
        # Restart from the initial model and fresh optimizer state (the
        # immutable-model equivalent of restore-on-exit).
        trial_model = model
        opt_state = optimizer.init(initial_params)
        lrs: list[float] = []
        losses: list[float] = []
        for i, lr in enumerate(lr_schedule):
            opt_state.hyperparams["learning_rate"] = lr  # ty: ignore[unresolved-attribute]
            inputs, targets = next(batches)
            key, step_key = jr.split(key)
            trial_model, opt_state, loss = test_lr(trial_model, opt_state, inputs, targets, step_key)
            loss = float(loss)

            emit_progress(
                zoom * steps_per_zoom + i + 1,
                total_steps,
                message=f"zoom {zoom + 1}/{num_zooms}",
            )

            if loss < min(losses, default=float("inf")):
                lrs.append(float(lr))
                losses.append(loss)

        if len(lrs) < 2:
            continue

        steepest_lr = _find_steepest(lrs, losses)
        lowest_lr = min(lowest_lr, _find_lowest_lr(lrs, losses))
        proposed_range = _propose_range(method, steepest_lr, lowest_lr)

        best_lr = cast(float, np.mean(proposed_range))
        history.append(LRFinderSeries(lrs=lrs, losses=losses, best_lr=best_lr, steepest_lr=steepest_lr, zoom=zoom + 1))

        current_range = _calculate_zoom_range(proposed_range, current_range, zoom_factor)

    if not np.isfinite(best_lr):
        raise RuntimeError("No valid learning rate found. Try increasing the range.")

    return best_lr, config, history


@validate_call
def _calculate_zoom_range(proposed_range: Range, current_range: Range, zoom_factor: float) -> Range:
    """Calculate the next learning rate range in log space."""
    log_start, log_end = np.log(current_range)
    log_low, log_high = np.log(proposed_range)

    new_log_start = log_start + (1 - zoom_factor) * (log_low - log_start)
    new_log_end = log_end - (1 - zoom_factor) * (log_end - log_high)

    return np.exp(new_log_start), np.exp(new_log_end)


@validate_call
def _get_lr_schedule(range: Range, n_steps: int) -> np.ndarray:
    start_lr, end_lr = range
    log_lrs = np.linspace(np.log(start_lr), np.log(end_lr), n_steps)
    return np.exp(log_lrs)


@validate_call
def _find_steepest(lrs: list[float], losses: list[float]) -> float:
    """Find best learning rate using gradient-weighted average."""
    lrs_array = np.array(lrs)
    losses_array = np.array(losses)

    gradients = losses_array[1:] - losses_array[:-1]
    weights = np.maximum(0, -(gradients - gradients.max()))

    if weights.sum() > 0:
        mid_points = np.exp((np.log(lrs_array[1:]) + np.log(lrs_array[:-1])) / 2)
        best_lr = np.exp(np.average(np.log(mid_points), weights=weights))
    else:
        best_lr = np.exp((np.log(lrs_array[0]) + np.log(lrs_array[-1])) / 2)

    return best_lr


@validate_call
def _find_lowest_lr(lrs: list[float], losses: list[float]) -> float:
    lrs_array = np.array(lrs)
    losses_array = np.array(losses)
    return lrs_array[np.argmin(losses_array)]


@validate_call
def _propose_range(method: SearchMethod, steepest_lr: float, lowest_lr: float) -> Range:
    if method == "balanced":
        return (steepest_lr, lowest_lr)
    elif method == "steepest":
        return (steepest_lr, steepest_lr)
    elif method == "lowest":
        return (lowest_lr, lowest_lr)
    else:
        raise ValueError("Unknown optimization method")

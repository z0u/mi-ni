from typing import Generator, TypeAlias, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.lr_finder.types import LRFinderConfig, LRFinderSeries, Progress, SearchMethod
from utils.param_types import validate_call
from utils.torch.mixed_precision import AMPContext
from utils.torch.training import mode, restore_state
from utils.torch.types import get_device

Range: TypeAlias = tuple[float, float]


@validate_call
def lr_finder_search(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    start_lr: float = 1e-10,
    end_lr: float = 1e1,
    num_zooms: int = 5,
    steps_per_zoom: int = 10,
    zoom_factor: float = 0.5,
    method: SearchMethod = 'steepest',
    amp_context: AMPContext | None = None,
):
    """Perform multi-scale learning rate range test with optional mixed precision."""
    if start_lr >= end_lr:
        raise ValueError('start_lr must be less than end_lr')

    if amp_context is None:
        amp_context = AMPContext(use_amp=False, device_type=get_device(model))

    yield Progress(step=0, total_steps=num_zooms * steps_per_zoom)
    yield LRFinderConfig(
        num_zooms=num_zooms,
        method=method,
        zoom_factor=zoom_factor,
        steps_per_zoom=steps_per_zoom,
        start_lr=start_lr,
        end_lr=end_lr,
    )

    best_lr = float('nan')
    steepest_lr = float('nan')
    lowest_lr = float('inf')

    data_loader = _cycle_loader(train_loader)
    current_range = (start_lr, end_lr)
    for zoom in range(num_zooms):
        yield Progress(step=zoom * steps_per_zoom, info={'zoom': zoom})

        lr_schedule = _get_lr_schedule(current_range, steps_per_zoom)
        # Test current range and find best LR
        with restore_state(model), restore_state(optimizer), mode(model, 'train'):
            lrs = []
            losses = []
            for i, lr in enumerate(lr_schedule):
                inputs, targets = next(data_loader)
                loss = _test_lr(
                    model,
                    criterion,
                    optimizer,
                    inputs,
                    targets,
                    lr,
                    amp_context,
                )

                yield Progress(step=zoom * steps_per_zoom + i, info={'zoom': zoom, 'step': i})

                # Store values if loss is stable
                if loss < min(losses, default=float('inf')):
                    lrs.append(lr)
                    losses.append(loss)

        if len(lrs) < 2:
            # Not enough stable data points to find a range at this zoom level
            continue

        # Local steepest descent, but global lowest loss
        steepest_lr = _find_steepest(lrs, losses)
        lowest_lr = min(lowest_lr, _find_lowest_lr(lrs, losses))
        proposed_range = _propose_range(method, steepest_lr, lowest_lr)

        # Store results and update visualization
        best_lr = cast(float, np.mean(proposed_range))
        yield LRFinderSeries(lrs=lrs, losses=losses, best_lr=best_lr, steepest_lr=steepest_lr, zoom=zoom + 1)

        # Calculate next range
        current_range = _calculate_zoom_range(proposed_range, current_range, zoom_factor)

    if not np.isfinite(best_lr):
        raise RuntimeError('No valid learning rate found. Try increasing the range.')
    yield best_lr


@validate_call
def _calculate_zoom_range(proposed_range: Range, current_range: Range, zoom_factor: float) -> Range:
    """Calculate the next learning rate range in log space."""
    log_start, log_end = np.log(current_range)
    log_low, log_high = np.log(proposed_range)

    # Calculate new range boundaries
    new_log_start = log_start + (1 - zoom_factor) * (log_low - log_start)
    new_log_end = log_end - (1 - zoom_factor) * (log_end - log_high)

    return np.exp(new_log_start), np.exp(new_log_end)


@validate_call
def _test_lr(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    lr: float,
    amp_context: AMPContext,
) -> float:
    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Forward pass

    with amp_context.forward_pass():
        outputs = model(inputs)

        # Handle transformer outputs
        if outputs.dim() == 3:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

        loss = criterion(outputs, targets)

    amp_context.backward_pass(loss, optimizer)

    return loss.item()


@validate_call
def _get_lr_schedule(range: Range, n_steps: int) -> np.ndarray:
    # Calculate learning rate schedule in log space
    start_lr, end_lr = range
    log_lrs = np.linspace(np.log(start_lr), np.log(end_lr), n_steps)
    return np.exp(log_lrs)


@validate_call
def _find_steepest(lrs: list[float], losses: list[float]) -> float:
    """Find best learning rate using gradient-weighted average."""
    # Convert to numpy arrays
    lrs_array = np.array(lrs)
    losses_array = np.array(losses)

    gradients = losses_array[1:] - losses_array[:-1]
    # Consider negative gradients, shifted to make all positive
    weights = np.maximum(0, -(gradients - gradients.max()))

    if weights.sum() > 0:
        # Use geometric mean of adjacent LRs, weighted by gradients
        mid_points = np.exp((np.log(lrs_array[1:]) + np.log(lrs_array[:-1])) / 2)
        best_lr = np.exp(np.average(np.log(mid_points), weights=weights))
    else:
        # Fallback to geometric mean of range if no negative gradients
        best_lr = np.exp((np.log(lrs_array[0]) + np.log(lrs_array[-1])) / 2)

    return best_lr


@validate_call
def _find_lowest_lr(lrs: list[float], losses: list[float]) -> float:
    lrs_array = np.array(lrs)
    losses_array = np.array(losses)
    return lrs_array[np.argmin(losses_array)]


@validate_call
def _propose_range(method: SearchMethod, steepest_lr: float, lowest_lr: float) -> Range:
    if method == 'balanced':
        return (steepest_lr, lowest_lr)
    elif method == 'steepest':
        return (steepest_lr, steepest_lr)
    elif method == 'lowest':
        return (lowest_lr, lowest_lr)
    else:
        raise ValueError('Unknown optimization method')


@validate_call
def _cycle_loader(loader: DataLoader) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    """Cycle through a DataLoader indefinitely."""
    iterator = iter(loader)
    while True:
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, targets = next(iterator)

        # Yield the batch
        yield inputs, targets

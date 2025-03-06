from contextlib import contextmanager
from copy import deepcopy
from typing import Literal

import torch


@contextmanager
def restore_state(model: torch.nn.Module | torch.optim.Optimizer):
    """Restore model and optimizer state on exit."""
    state = deepcopy(model.state_dict())
    try:
        yield
    finally:
        # Restore model and optimizer state
        model.load_state_dict(state)


@contextmanager
def mode(model: torch.nn.Module, mode: Literal['train', 'eval']):
    """Set model to training or eval mode, and restore initial mode on exit."""
    state = model.training
    model.train(mode=mode == 'train')
    try:
        yield
    finally:
        model.train(state)

import logging
from contextlib import contextmanager

import torch

from utils.torch.types import get_dtype, supports_bfloat16

log = logging.getLogger(__name__)


class AMPContext:
    # These are defined below so they can be pickled separately
    use_amp: bool
    device_type: str
    dtype: torch.dtype
    scaler: torch.GradScaler | None

    def __init__(
        self,
        use_amp: bool = False,
        device_type: str = 'cpu',
        dtype: torch.dtype | str | None = None,
        scaler: torch.GradScaler | None = None,
    ):
        self.use_amp = use_amp
        self.device_type = device_type
        if dtype is None:
            self.dtype = (
                (torch.bfloat16 if supports_bfloat16(device_type) else torch.float16) if use_amp else torch.float32
            )
        elif isinstance(dtype, str):
            self.dtype = get_dtype(dtype)

        if use_amp and scaler is None:
            scaler = torch.GradScaler(device=device_type)
        self.scaler = scaler

        self.first_autocast = True
        self.first_backward = True

    @contextmanager
    def forward_pass(self):
        """Context manager for mixed precision, if enabled."""
        if self.use_amp and self.first_autocast:
            log.info(f'Using mixed precision for forward pass with dtype {self.dtype}')
            self.first_autocast = False

        with torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.use_amp) as autocast:
            yield autocast

    def backward_pass(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        """
        Backward pass with mixed precision, if enabled.

        Runs the backward pass and optimizer step, using gradient scaling if
        enabled. The scaler is used to scale the loss before the backward pass,
        and the optimizer step is performed using the scaled gradients.

        Args:
            loss (torch.Tensor): The loss tensor.
            optimizer (torch.optim.Optimizer): The optimizer to step.
        """
        if self.scaler and self.first_backward:
            log.info('Using gradient scaler for backward pass')
            self.first_backward = False

        optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()

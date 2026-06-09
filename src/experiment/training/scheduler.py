import warnings

from pydantic import NonNegativeInt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from experiment.config import SchedulerConfig


def configure_scheduler(optimizer: Optimizer, config: SchedulerConfig, epoch_length: NonNegativeInt) -> LRScheduler:
    """Linear warmup to the peak LR, then cosine anneal down to `min_lr_factor * peak`.

    The scheduler steps once per batch (see `GPTModule.configure_optimizers`), so all
    durations are expressed in steps rather than epochs.
    """
    warnings.filterwarnings('ignore', message=r'.*The epoch parameter in.*scheduler.step', category=UserWarning)

    total_steps = config.epochs * epoch_length
    warmup_steps = int(config.warmup_epochs * epoch_length)
    base_lr = optimizer.param_groups[0]['lr']
    eta_min = config.min_lr_factor * base_lr

    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min)
    if warmup_steps == 0:
        return cosine

    warmup = LinearLR(optimizer, start_factor=config.min_lr_factor, total_iters=warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

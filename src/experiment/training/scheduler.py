import warnings

from pydantic import NonNegativeInt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler

from experiment.config import SchedulerConfig


def configure_scheduler(optimizer: Optimizer, config: SchedulerConfig, epoch_length: NonNegativeInt) -> LRScheduler:
    warnings.filterwarnings('ignore', message=r'.*The epoch parameter in.*scheduler.step', category=UserWarning)

    # Create schedulers for each phase
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=config.min_lr_factor,
        total_iters=int(config.warmup_epochs * epoch_length),
    )
    return warmup_scheduler

    # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=0.1)

    # # Combine them sequentially
    # return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

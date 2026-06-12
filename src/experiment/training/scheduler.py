import optax
from pydantic import NonNegativeInt

from experiment.config import SchedulerConfig


def configure_schedule(config: SchedulerConfig, peak_lr: float, epoch_length: NonNegativeInt) -> optax.Schedule:
    """Linear warmup to the peak LR, then cosine anneal down to `min_lr_factor * peak`.

    The schedule is evaluated once per batch (see the training loop), so all
    durations are expressed in steps rather than epochs.
    """
    total_steps = config.epochs * epoch_length
    warmup_steps = int(config.warmup_epochs * epoch_length)
    min_lr = config.min_lr_factor * peak_lr

    return optax.warmup_cosine_decay_schedule(
        init_value=min_lr,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=min_lr,
    )

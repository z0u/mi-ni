"""
A toy experiment for exercising the detached run lifecycle on local compute.

Launch it with::

    python -m mini launch experiments/toy.py --workers 3

It sweeps a "learning rate" over three jobs. The largest LR diverges on purpose,
so you can watch a failure surface in `status`, read its traceback with `logs`,
and re-run it with `retry`.
"""

from __future__ import annotations

import random
import time

from mini import Experiment, emit_metrics, emit_progress, get_data_dir


def train(lr: float, steps: int = 10) -> dict:
    """Pretend to train: emit progress + metrics, persist an artifact, return a summary."""
    rng = random.Random(round(lr * 1e6))
    loss = 10.0
    for i in range(steps):
        time.sleep(0.2)
        # A high learning rate diverges; everything else descends.
        loss = loss * (0.88 + rng.uniform(-0.02, 0.02)) if lr <= 5e-2 else loss * 3.0
        emit_progress(i + 1, steps, message=f'lr={lr:g}')
        emit_metrics(loss=round(loss, 4), lr=lr)
        if loss > 1e4:
            raise RuntimeError(f'diverged at step {i + 1} (lr={lr:g} too high)')

    (get_data_dir() / f'loss_{lr:g}.txt').write_text(f'{loss:.4f}\n')
    return {'lr': lr, 'final_loss': round(loss, 4)}


experiment = Experiment(
    name='toy',
    fn=train,
    configs=[(1e-3,), (1e-2,), (2e-1,)],  # the last one diverges
)

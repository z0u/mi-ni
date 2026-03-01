"""Lightning callbacks that integrate with mini's progress infrastructure."""

from typing import override

import lightning as L

from mini.progress import emit_progress


class LightningProgress(L.Callback):
    """Report training progress via emit_progress.

    Drop this into any ``L.Trainer`` to get mini-compatible progress reporting —
    works transparently whether the training function runs locally or remotely.

    Example::

        trainer = L.Trainer(
            callbacks=[LightningProgress()],
            enable_progress_bar=False,
        )
    """

    @override
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx: int):
        total = int(trainer.estimated_stepping_batches)
        step = trainer.global_step
        loss = trainer.callback_metrics.get('train_loss')
        message = f'loss={float(loss):.4f}' if loss is not None else ''
        emit_progress(step, total, message=message)

"""Tests for the Lightning -> mini progress integration."""

from __future__ import annotations

import queue

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mini.progress import ProgressMessage, progress_context
from mini.torch.lightning import LightningProgress


class _TinyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.layer(x), y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _loader(n: int = 4) -> DataLoader:
    x = torch.randn(n, 2)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=1)


def _drain(q: queue.Queue) -> list[ProgressMessage]:
    out: list[ProgressMessage] = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_lightning_progress_emits_messages_to_queue():
    """Callback emits ProgressMessage updates with correct step/total."""
    q: queue.Queue = queue.Queue()
    total_steps = 4
    # Use a long debounce interval so flush at context exit is what delivers
    # the trailing message — this verifies the flush hookup, not just timing.
    with progress_context('run-1', 'job-1', queue=q, emission_interval=10.0):
        trainer = L.Trainer(
            max_steps=total_steps,
            callbacks=[LightningProgress()],
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            accelerator='cpu',
            devices=1,
        )
        trainer.fit(_TinyModel(), _loader(total_steps))

    messages = _drain(q)
    assert messages, 'expected at least one progress message'
    assert all(isinstance(m, ProgressMessage) for m in messages)
    assert all(m.run_id == 'run-1' and m.job_id == 'job-1' for m in messages)
    assert all(m.total == total_steps for m in messages)
    # Steps should be monotonic and within range
    steps = [m.step for m in messages]
    assert steps == sorted(steps)
    assert max(steps) == total_steps - 1 or max(steps) == total_steps
    # The latest emission should carry a loss message
    assert 'loss=' in messages[-1].message


def test_lightning_progress_is_inert_outside_job_context(capsys):
    """Without a progress_context the callback falls back to print, not crash."""
    trainer = L.Trainer(
        max_steps=1,
        callbacks=[LightningProgress()],
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        accelerator='cpu',
        devices=1,
    )
    trainer.fit(_TinyModel(), _loader(1))
    # No exception is the main assertion. emit_progress() returns silently
    # when there's no JobContext, so nothing relating to a ProgressMessage
    # URN should hit stdout either.
    captured = capsys.readouterr()
    assert 'mini:run:' not in captured.out


@pytest.mark.parametrize('total', [2, 5])
def test_lightning_progress_total_matches_estimated_batches(total):
    """`total` reported equals trainer.estimated_stepping_batches."""
    q: queue.Queue = queue.Queue()
    with progress_context('r', 'j', queue=q, emission_interval=10.0):
        trainer = L.Trainer(
            max_steps=total,
            callbacks=[LightningProgress()],
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            accelerator='cpu',
            devices=1,
        )
        trainer.fit(_TinyModel(), _loader(total))

    messages = _drain(q)
    assert messages
    assert {m.total for m in messages} == {total}

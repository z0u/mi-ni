"""End-to-end training: the loop runs, checkpoints, and reports progress."""

import math
import queue

import equinox as eqx
import numpy as np
import pytest

from experiment.compute.data_pipelines import save_data
from experiment.compute.model import load_checkpoint
from experiment.compute.training import train_model
from experiment.config import (
    CorpusMetadata,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from mini.progress import ProgressMessage, progress_context

VOCAB = [chr(ord("a") + i) for i in range(26)]


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(1, len(VOCAB) + 1, size=10_000).astype(np.int32)
    metadata = CorpusMetadata(
        tokenizer_config=TokenizerConfig(vocabulary=VOCAB),
        total_tokens=len(data),
        total_chars=len(data),
        sources=[],
    )
    save_data(data, metadata, tmp_path)
    return tmp_path


def make_training_config(dropout: float = 0.1, **model_overrides) -> TrainingConfig:
    return TrainingConfig(
        model=ModelConfig(
            vocab_size=64,
            block_size=64,
            n_embd=32,
            n_head=8,
            n_head_dim=8,
            n_ff=32,
            n_layer=1,
            dropout=dropout,
            **model_overrides,
        ),
        tokenizer=TokenizerConfig(vocabulary=VOCAB),
        data=DataConfig(batch_size=8, oversample=1, train_split=0.8, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=1e-3, learning_rate=1e-3, betas=(0.9, 0.95)),
        scheduler=SchedulerConfig(epochs=2, warmup_epochs=1, min_lr_factor=0.01),
    )


@pytest.mark.parametrize("arch", ["gpt", "ngpt"])
def test_train_model_end_to_end(data_dir, arch):
    """Training produces per-epoch metrics and a checkpoint equivalent to the returned model."""
    # nGPT is dropout-free (the unit-hypersphere constraint regularizes instead).
    config = make_training_config(architecture=arch, dropout=0 if arch == "ngpt" else 0.1)
    model, metrics = train_model(config, data_dir)

    assert [m.epoch for m in metrics] == [0, 1]
    assert all(math.isfinite(m.val_loss) for m in metrics)
    # Random tokens: loss should be in the vicinity of ln(vocab), not diverged.
    assert metrics[-1].val_loss < 2 * math.log(config.model.vocab_size)

    loaded, loaded_config, loaded_metrics = load_checkpoint(data_dir)
    assert loaded_config.model.architecture == arch
    assert loaded_metrics is not None and loaded_metrics.epoch == 1

    idx = np.tile(np.arange(16), (2, 1))
    model, loaded = eqx.nn.inference_mode(model), eqx.nn.inference_mode(loaded)
    np.testing.assert_allclose(loaded(idx), model(idx), rtol=0, atol=0)


def test_progress_emitted_during_training(data_dir):
    """The training loop reports step progress through mini's progress context."""
    q: queue.Queue = queue.Queue()
    config = make_training_config()
    # Long debounce interval: the flush at context exit delivers the trailing message.
    with progress_context("run-1", "job-1", queue=q, emission_interval=10.0):
        train_model(config, data_dir)

    messages: list[ProgressMessage] = []
    while not q.empty():
        messages.append(q.get_nowait())

    assert messages, "expected at least one progress message"
    assert all(m.run_id == "run-1" and m.job_id == "job-1" for m in messages)
    steps = [m.step for m in messages]
    assert steps == sorted(steps)
    assert {m.total for m in messages} == {max(steps)}, "final step should equal the reported total"
    assert "loss=" in messages[-1].message

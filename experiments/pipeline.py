"""
A gpt_sweep-shaped multi-step experiment on local compute.

One data-prep step, then a training sweep whose configs depend on prep's output.
Drive it across wakes (each call advances the DAG and returns immediately):

    python -m mini run experiments/pipeline.py     # launches prep, suspends
    python -m mini run experiments/pipeline.py      # ...prep done -> launches sweep
    python -m mini run experiments/pipeline.py      # ...until ✓ complete

Re-running only ever executes the un-run / failed pieces — memoized by content,
so a crash heals by re-running, and editing a step re-runs just that step.
"""

from __future__ import annotations

import time

from mini import Ctx, Experiment, emit_metrics, emit_progress, get_data_dir


def prepare_data() -> dict:
    """Pretend to download + tokenize a corpus; write it to the volume."""
    time.sleep(1.0)
    text = 'the quick brown fox jumps over the lazy dog ' * 200
    (get_data_dir() / 'corpus.txt').write_text(text)
    return {'vocab_size': len(set(text)), 'n_chars': len(text)}


def train(lr: float, vocab_size: int) -> dict:
    """Train one config; depends on prep's vocab_size (like gpt_sweep)."""
    loss = 5.0
    for step in range(8):
        time.sleep(0.2)
        loss *= 0.85
        emit_progress(step + 1, 8, message=f'lr={lr:g}')
        emit_metrics(loss=round(loss, 4), lr=lr, vocab=vocab_size)
    return {'lr': lr, 'val_loss': round(loss, 4)}


def main(ctx: Ctx) -> dict:
    meta = ctx.run(prepare_data)  # single prep step; suspends until done
    vocab = meta['vocab_size']  # the dependency the single-map model can't express
    results = ctx.map(train, [(lr, vocab) for lr in (1e-3, 1e-2, 1e-1)])
    best = min(results, key=lambda r: r['val_loss'])
    return {'meta': meta, 'best': best, 'results': results}


experiment = Experiment(name='pipeline', main=main)

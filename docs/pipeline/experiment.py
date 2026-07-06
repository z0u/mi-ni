"""
A minimal multi-step experiment: one data-prep step, then a training sweep whose
configs depend on prep's output. This is the *definition* — plain, importable
Python with no notebook or compute baked in. The companion ``report.py`` reads
its durable results and renders them; the ``experiment`` skill explains the flow.

Drive it to completion with a live progress bar:

    bin/mini run docs/pipeline/experiment.py --watch --workers 3

``--watch`` ticks the DAG to launch each stage, then polls the durable records
and renders a bar per task until done. Ctrl-C only stops *watching* — the task
workers are detached, so re-running the same command resumes monitoring (done
steps are memo hits; in-flight tasks aren't relaunched).

Or drive it one wake at a time (each call advances the DAG and returns at once):

    bin/mini run docs/pipeline/experiment.py     # launches prep, suspends
    bin/mini run docs/pipeline/experiment.py     # ...prep done -> launches sweep
    bin/mini run docs/pipeline/experiment.py     # ...until ✓ complete

Re-running only ever executes the un-run / failed pieces — memoized by content,
so a crash heals by re-running, and editing a step re-runs just that step.
"""

from __future__ import annotations

import math
import time

from mini import Ctx, Experiment, emit_metrics, emit_progress, get_data_dir


def prepare_data() -> dict:
    """Pretend to download + tokenize a corpus; write it to the volume."""
    time.sleep(1.0)
    text = "the quick brown fox jumps over the lazy dog " * 200
    (get_data_dir() / "corpus.txt").write_text(text)
    return {"vocab_size": len(set(text)), "n_chars": len(text)}


def train(lr: float, vocab_size: int) -> dict:
    """Train one config. Keyed on (lr, vocab_size) — pass the *narrow* subset a
    task actually uses, so an unrelated config change doesn't bust its memo.
    """
    # A toy loss bowl: descent is fastest near lr=1e-2; too small is slow to
    # converge, too large overshoots. So the sweep has a real best.
    quality = math.exp(-((math.log10(lr) - math.log10(1e-2)) ** 2))
    loss = 5.0
    for step in range(8):
        time.sleep(0.2)
        loss *= 1 - 0.3 * quality
        emit_progress(step + 1, 8, message=f"lr={lr:g}")
        emit_metrics(loss=round(loss, 4), lr=lr, vocab=vocab_size)
    return {"lr": lr, "val_loss": round(loss, 4)}


def main(ctx: Ctx) -> dict:
    # The body is plain Python and re-runs every wake, so keep it cheap and
    # deterministic: derive configs here, do the heavy/non-deterministic work
    # inside tasks (with any seed folded into their inputs).
    meta = ctx.run(prepare_data)  # single prep step; suspends until done
    vocab = meta["vocab_size"]  # the dependency a single-map experiment can't express
    lrs = [1e-3, 1e-2, 1e-1]
    results = ctx.map(train, lrs, [vocab] * len(lrs))  # zipped, Executor-style: train(lr, vocab)
    best = min(results, key=lambda r: r["val_loss"])
    return {"meta": meta, "best": best, "results": results}


experiment = Experiment(name="pipeline", main=main)

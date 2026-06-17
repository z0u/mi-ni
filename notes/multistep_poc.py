"""
Validate the memoized, re-runnable orchestration model for multi-step experiments
(the gpt_sweep shape: a single prep step, then a sweep that depends on its output).

The experiment is a plain function ``main(ctx)`` that expresses the DAG in
ordinary Python. Each ``ctx.run`` / ``ctx.map`` is content-addressed (keyed by
the function's *source* + its inputs):

  - cached (DONE)  -> return the stored result instantly
  - in flight      -> raise Pending (suspend this wake)
  - absent/FAILED  -> launch a detached worker, then raise Pending

The driver re-runs ``main`` every "wake". Completed steps are memo hits; only the
un-run / failed pieces execute. So crash-recovery == re-run the whole thing.
A worker dying, or the launcher dying, both heal on the next wake.

Run the whole demo with:  python notes/multistep_poc.py demo
"""

from __future__ import annotations

import hashlib
import inspect
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import cloudpickle

STORE = Path(os.environ.get('POC_STORE', '/tmp/mini_multistep/memo'))


class Pending(Exception):
    """Raised to suspend the driver until in-flight work finishes."""


# --- content-addressed memo -------------------------------------------------
def _key(fn, args: tuple) -> str:
    # Hash the function *source* (so fixing a bug invalidates the memo) + inputs.
    src = inspect.getsource(fn)
    return hashlib.sha1(src.encode() + repr(args).encode()).hexdigest()[:12]


def _state(d: Path) -> str:
    s = d / 'state'
    return s.read_text() if s.exists() else 'absent'


def _launch(fn, args: tuple, key: str) -> None:
    d = STORE / key
    d.mkdir(parents=True, exist_ok=True)
    (d / 'call.pkl').write_bytes(cloudpickle.dumps((fn, args)))
    (d / 'state').write_text('running')
    subprocess.Popen(
        [sys.executable, __file__, 'worker', key],
        start_new_session=True,  # outlive the driver (local stand-in for detach)
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class Ctx:
    """The memoized, non-blocking run/map the orchestration calls."""

    def __init__(self):
        self.waiting = 0

    def run(self, fn, *args):
        key = _key(fn, args)
        d = STORE / key
        st = _state(d)
        if st == 'done':
            return pickle.loads((d / 'result.pkl').read_bytes())
        if st in ('absent', 'failed'):
            _launch(fn, args, key)
        self.waiting += 1
        return _Hole()  # never actually used; we suspend below

    def map(self, fn, items: list[tuple]):
        results, missing = [], False
        for args in items:
            key = _key(fn, args)
            d = STORE / key
            st = _state(d)
            if st == 'done':
                results.append(pickle.loads((d / 'result.pkl').read_bytes()))
            else:
                if st in ('absent', 'failed'):
                    _launch(fn, args, key)  # launch all missing -> they run in parallel
                missing = True
        if missing:
            self.waiting += len(items) - len(results)
            raise Pending(f'{len(items) - len(results)}/{len(items)} sweep jobs not done')
        return results


class _Hole:
    """Returned by ctx.run when not ready; touching it suspends the wake."""

    def __getattr__(self, _):
        raise Pending('upstream step not done')

    def __getitem__(self, _):
        raise Pending('upstream step not done')


# --- the worker (runs one memoized task) ------------------------------------
def worker(key: str) -> None:
    d = STORE / key
    fn, args = cloudpickle.loads((d / 'call.pkl').read_bytes())
    try:
        result = fn(*args)
        (d / 'result.pkl').write_bytes(pickle.dumps(result))
        (d / 'state').write_text('done')
    except Exception as e:
        (d / 'error.txt').write_text(repr(e))
        (d / 'state').write_text('failed')


# --- the experiment: a plain DAG, gpt_sweep-shaped --------------------------
def prepare_data():
    time.sleep(1.0)
    return {'vocab_size': 64, 'n_train': 10_000}  # "tokenizer metadata"


def train(config: dict):
    # One config hits a transient failure on its first attempt, then recovers.
    store = Path(os.environ['POC_STORE'])
    attempt_file = store / f'.attempts_{config["lr"]}'
    n = int(attempt_file.read_text()) if attempt_file.exists() else 0
    attempt_file.write_text(str(n + 1))
    time.sleep(0.8)
    if config['lr'] == 1e-2 and n == 0:
        raise RuntimeError('transient worker crash')
    return {'lr': config['lr'], 'val_loss': round(0.5 + config['lr'], 4)}


def main(ctx: Ctx):
    # The dependency the current single-map model can't express:
    meta = ctx.run(prepare_data)
    vocab = meta['vocab_size']  # touching meta suspends until prep is DONE
    configs = [{'lr': lr, 'vocab_size': vocab} for lr in (1e-3, 1e-2, 1e-1)]
    results = ctx.map(train, [(c,) for c in configs])
    return meta, results


# --- driver: one wake = re-run main() to completion-or-suspend --------------
def tick() -> bool:
    ctx = Ctx()
    try:
        meta, results = main(ctx)
    except Pending as p:
        print(f'  · suspended ({p})')
        return False
    print('  ✓ COMPLETE')
    print(f'    prep: {meta}')
    for r in results:
        print(f'    {r}')
    return True


def demo() -> None:
    import shutil

    shutil.rmtree(STORE.parent, ignore_errors=True)
    STORE.mkdir(parents=True, exist_ok=True)
    for i in range(1, 40):
        print(f'wake #{i}:')
        if tick():
            break
        time.sleep(0.4)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'worker':
        worker(sys.argv[2])
    elif cmd == 'demo':
        demo()
    elif cmd == 'tick':
        tick()

"""Snapshot a finished ``gpt-sweep`` run's val-loss curves to ``results.json``.

The report (``report.py``) reads this committed file so it opens standalone — no
Modal, no GPU, no waiting. Run it once the experiment has completed:

    bin/mini run docs/gpt-sweep/experiment.py --app modal --max-containers 9
    uv run docs/gpt-sweep/snapshot.py            # reads the Modal run by default

Each ``train_one`` cell returns ``(arch_label, lr_str, [val_loss per epoch])``;
this collects the DONE cells into ``{f'{arch}|{lr}': [losses]}`` — the shape
``report.py`` expects — and writes ``results.json`` beside it. Re-running after a
``retry`` just refreshes the snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini.runs import RunState

NAME = 'gpt-sweep'
RESULTS_PATH = Path(__file__).resolve().parent / 'results.json'


def _memo_store(app: str):
    """The memo store for the run on the chosen backend (reads only, no relaunch)."""
    if app == 'local':
        from mini.memo import MemoStore
        from mini.runs import data_root

        return MemoStore(data_root() / NAME)
    from mini.modal_apparatus import ModalApparatus

    return ModalApparatus(NAME).memo_store()


def main() -> None:
    parser = argparse.ArgumentParser(description='Snapshot gpt-sweep val-loss curves to results.json')
    parser.add_argument('--app', choices=['local', 'modal'], default='modal', help='backend the run executed on')
    args = parser.parse_args()

    store = _memo_store(args.app)
    curves: dict[str, list[float]] = {}
    for rec in store.records():
        if rec.get('state') != RunState.DONE:  # skip in-flight / failed cells (and unsettled prep)
            continue
        result = store.result(rec['key'])
        if isinstance(result, tuple) and len(result) == 3:  # a train_one cell, not prep's metadata
            arch, lr, losses = result
            curves[f'{arch}|{lr}'] = list(losses)

    if not curves:
        raise SystemExit(f'no completed train cells for {NAME!r} on --app {args.app} — run the experiment first')

    RESULTS_PATH.write_text(json.dumps(curves, indent=2) + '\n')
    print(f'wrote {len(curves)} curves to {RESULTS_PATH.relative_to(Path.cwd())}')


if __name__ == '__main__':
    main()

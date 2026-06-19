"""Minimal role-selection demo: prove a step's role routes it to the right hardware.

No real work — each step just reports the GPU it landed on (via ``nvidia-smi``,
which Modal injects into GPU containers). The ``probe`` role is CPU-only, ``gpu``
asks for an L4, so on Modal the records show one step saw no GPU and the other an L4.

    bin/mini run docs/role-demo/experiment.py --app modal
    bin/mini results role-demo --app modal

Note the ``label`` argument: the memo key is ``fingerprint(fn, args)`` and excludes
the hardware, so calling the same fn with the same args twice would be a memo *hit*
(both steps returning the first's result). The label keeps the two keys distinct.
"""

from __future__ import annotations

from mini import Ctx, Experiment


def probe_hardware(label: str) -> str:
    """Report the GPU this task landed on, or 'cpu' if there's none."""
    import shutil
    import subprocess

    if not shutil.which('nvidia-smi'):
        return 'cpu'
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() or 'cpu'


def main(ctx: Ctx) -> dict[str, str]:
    on_cpu = ctx.run(probe_hardware, 'probe', role='probe')
    on_gpu = ctx.run(probe_hardware, 'gpu', role='gpu')
    return {'probe': on_cpu, 'gpu': on_gpu}


experiment = Experiment(
    name='role-demo',
    main=main,
    roles={
        'probe': {},  # CPU-only
        'gpu': dict(gpu='L4', timeout=120),
    },
)

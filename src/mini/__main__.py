"""
``python -m mini`` — drive detached experiments across short-lived processes.

Each subcommand is a quick, stateless call against the durable run state, so an
agent (or you) can launch, poll, fix, and gather without holding a session open:

    python -m mini launch experiments/toy.py   # → toy/3f9a   (detached)
    python -m mini ls                            # experiments + their runs
    python -m mini status toy                    # latest run, by NAME
    python -m mini status toy/3f9a               # a specific run
    python -m mini logs   toy/3f9a --job 2       # full traceback
    python -m mini retry  toy/3f9a --failed      # re-run failed jobs
    python -m mini cancel toy/3f9a
    python -m mini results toy/3f9a
"""

from __future__ import annotations

import argparse
import time

from mini.experiment import load_experiment
from mini.runs import DATA_ROOT, RunState, Run, latest_run, open_experiment, open_run

_GLYPH = {
    RunState.PENDING: '·',
    RunState.RUNNING: '▸',
    RunState.DONE: '✓',
    RunState.FAILED: '✗',
    RunState.CANCELLED: '⊘',
}


def _resolve(ref: str) -> Run:
    """A *ref* is ``<experiment>`` (latest run) or ``<experiment>/<token>``."""
    if '/' in ref:
        return open_run(ref)
    run = latest_run(ref)
    if run is None:
        raise SystemExit(f'no runs found for experiment {ref!r}')
    return run


def _fmt_metrics(metrics: dict[str, float]) -> str:
    return '  '.join(f'{k}={v:g}' for k, v in metrics.items())


def _age(ts: float | None) -> str:
    return f'{time.time() - ts:.0f}s ago' if ts else '—'


def cmd_launch(args: argparse.Namespace) -> None:
    print(load_experiment(args.path).submit().id)


def cmd_ls(args: argparse.Namespace) -> None:
    root = DATA_ROOT
    names = sorted(p.name for p in root.glob('*') if (p / '.control' / 'index.json').exists()) if root.exists() else []
    if not names:
        print('no experiments yet (nothing under .mini/*/.control)')
        return
    for name in names:
        _, tokens = open_experiment(name)
        print(f'{name}  ({len(tokens)} run{"s" * (len(tokens) != 1)})')
        for token in tokens[:5]:
            run = open_run(f'{name}/{token}')
            states = [s.state for s in run.status()]
            done = sum(s == RunState.DONE for s in states)
            print(f'    {token}  {run.state():9}  {done}/{len(states)} done')


def cmd_status(args: argparse.Namespace) -> None:
    run = _resolve(args.ref)
    statuses = run.status()
    print(f'{run.id}  —  {run.state()}  ({len(statuses)} jobs)')
    for s in statuses:
        line = f'  {_GLYPH.get(s.state, "?")} job {s.job_id}  {s.state:9}  {s.step}/{s.total}'
        if s.metrics:
            line += f'  {_fmt_metrics(s.metrics)}'
        if s.message:
            line += f'  — {s.message}'
        if s.state == RunState.RUNNING:
            line += f'  ♥ {_age(s.heartbeat_at)}'
        if s.error:
            line += f'  !! {s.error}'
        print(line)


def cmd_results(args: argparse.Namespace) -> None:
    run = _resolve(args.ref)
    try:
        for value in run.results():
            print(value)
    except RuntimeError as e:
        raise SystemExit(str(e)) from e


def cmd_logs(args: argparse.Namespace) -> None:
    print(_resolve(args.ref).logs(args.job))


def cmd_retry(args: argparse.Namespace) -> None:
    run = _resolve(args.ref).retry(failed_only=args.failed)
    print(f'relaunched {run.id}')


def cmd_cancel(args: argparse.Namespace) -> None:
    _resolve(args.ref).cancel()
    print(f'cancelled {args.ref}')


def main() -> None:
    parser = argparse.ArgumentParser(prog='mini', description='Drive detached mi-ni experiments.')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('launch', help='import an experiment file and launch it detached')
    p.add_argument('path')
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser('ls', help='list experiments and their runs')
    p.set_defaults(func=cmd_ls)

    for name, help_text in [('status', 'show per-job state + metrics'), ('results', 'gather results')]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument('ref', help='<experiment> (latest run) or <experiment>/<token>')
        p.set_defaults(func={'status': cmd_status, 'results': cmd_results}[name])

    p = sub.add_parser('logs', help='print a job traceback/log')
    p.add_argument('ref')
    p.add_argument('--job', required=True)
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser('retry', help='relaunch failed (or all unfinished) jobs')
    p.add_argument('ref')
    p.add_argument('--failed', action='store_true', help='only re-run FAILED jobs (default: all unfinished)')
    p.set_defaults(func=cmd_retry)

    p = sub.add_parser('cancel', help='stop a run and mark unfinished jobs cancelled')
    p.add_argument('ref')
    p.set_defaults(func=cmd_cancel)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

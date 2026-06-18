"""
``python -m mini`` — run and monitor memoized experiments across short-lived processes.

An experiment is a ``main(ctx)`` DAG (or a single sweep, lowered to one ``ctx.map``).
Each subcommand is a quick, stateless call against the durable memo store, so an
agent (or you) can drive, poll, and gather without holding a session open:

    python -m mini run    experiments/pipeline.py --watch  # drive a DAG to completion (live bar)
    python -m mini run    experiments/pipeline.py          # advance one wake, then return
    python -m mini ls                                       # experiments + task state
    python -m mini status pipeline                          # per-task state + metrics, by NAME
    python -m mini results pipeline                         # per-task results
    python -m mini logs   pipeline <key>                    # a failed task's traceback
    python -m mini cancel pipeline                          # stop in-flight tasks

State is addressed by experiment NAME (one memo store per experiment). Read
commands take ``--app modal`` to inspect a run on the Modal control plane.
"""

from __future__ import annotations

import argparse
import time

from mini.apparatus import Apparatus
from mini.experiment import load_experiment
from mini.local_apparatus import LocalApparatus
from mini.memo import MemoStore
from mini.orchestration import tick
from mini.runs import DATA_ROOT, SETTLED, RunState

_GLYPH = {
    RunState.PENDING: '·',
    RunState.RUNNING: '▸',
    RunState.DONE: '✓',
    RunState.FAILED: '✗',
    RunState.CANCELLED: '⊘',
}


def _build_apparatus(name: str, args: argparse.Namespace) -> Apparatus:
    """Construct the apparatus the experiment runs on, from CLI flags.

    Compute is an execution choice, not part of the experiment definition.
    """
    backend = getattr(args, 'app', 'local')
    if backend == 'local':
        return LocalApparatus(name, max_workers=getattr(args, 'workers', 1))
    if backend == 'modal':
        from mini.modal_apparatus import ModalApparatus

        return ModalApparatus(name)
    raise SystemExit(f'--app {backend!r} not supported (use "local" or "modal")')


def _store_for(name: str, args: argparse.Namespace) -> MemoStore:
    """The memo store for an experiment by name, on the selected backend.

    Local reads straight off disk (no apparatus needed); ``--app modal`` builds
    the apparatus so reads hit the Modal control plane (a named ``modal.Dict``).
    """
    if getattr(args, 'app', 'local') == 'local':
        return MemoStore(DATA_ROOT / name)
    return _build_apparatus(name, args).memo_store()


def _fmt_metrics(metrics: dict[str, float]) -> str:
    return '  '.join(f'{k}={v:g}' for k, v in metrics.items())


def _age(ts: float | None) -> str:
    return f'{time.time() - ts:.0f}s ago' if ts else '—'


def _aggregate_state(states: list[RunState]) -> RunState:
    """Roll per-task states up to one experiment state."""
    if not states or all(s == RunState.DONE for s in states):
        return RunState.DONE
    if all(s in SETTLED for s in states):
        return RunState.CANCELLED if any(s == RunState.CANCELLED for s in states) else RunState.FAILED
    return RunState.RUNNING


def _rec_state(rec: dict) -> RunState:
    return RunState(rec['state']) if rec.get('state') else RunState.PENDING


def _memo_line(rec: dict) -> str:
    """One status line for a memoized task record (shared by `run`/`status`)."""
    state = _rec_state(rec)
    line = f'  {_GLYPH.get(state, "?")} {rec.get("fn", "task"):14} {rec["key"]:26} {state:9}'
    if rec.get('total'):
        line += f'  {rec.get("step", 0)}/{rec["total"]}'
    if rec.get('metrics'):
        line += f'  {_fmt_metrics(rec["metrics"])}'
    if state == RunState.RUNNING and rec.get('heartbeat_at'):
        line += f'  ♥ {_age(rec["heartbeat_at"])}'
    if rec.get('fc_id'):
        line += f'  [{rec["fc_id"]}]'  # Modal FunctionCall id — for log lookup / liveness
    if rec.get('error'):
        line += f'  !! {rec["error"]}'
    return line


def cmd_run(args: argparse.Namespace) -> None:
    """One wake of a (possibly multi-step) orchestration: advance + report.

    With ``--watch``, instead drive the DAG to completion with a live progress
    bar; Ctrl-C stops watching (detached workers live on — re-run to resume).
    """
    exp = load_experiment(args.path)
    apparatus = _build_apparatus(exp.name, args)
    if args.watch:
        _watch(exp, apparatus, poll=args.poll)
        return
    done, payload = tick(exp, apparatus)
    print(f'{exp.name}:')
    for rec in apparatus.memo_store().records():
        print(_memo_line(rec))
    if done:
        print(f'✓ complete: {payload}')
    else:
        print(f'… suspended — {payload} (re-run to advance)')


def _watch(exp, apparatus: Apparatus, poll: float) -> None:
    """Drive an orchestration to completion with a live bar (the ``--watch`` path)."""
    from mini.monitor import ExperimentFailed, drive_and_watch

    try:
        payload = drive_and_watch(exp, apparatus, poll=poll)
    except KeyboardInterrupt:
        print('\n… stopped watching; tasks keep running. Re-run the same command to resume.')
        return
    except ExperimentFailed as e:
        print(f'✗ {e}')
        for rec in e.failed:
            print(f'  ✗ {rec["key"]}  !! {rec.get("error", "")}')
        print(f'inspect a traceback with:  python -m mini logs {exp.name} <key>')
        raise SystemExit(1) from e
    print(f'✓ complete: {payload}')


def cmd_ls(args: argparse.Namespace) -> None:
    root = DATA_ROOT
    names = sorted(p.name for p in root.glob('*') if (p / '.control' / 'memo').is_dir()) if root.exists() else []
    if not names:
        print('no experiments yet (run one with: python -m mini run <path>)')
        return
    for name in names:
        states = [_rec_state(r) for r in MemoStore(DATA_ROOT / name).records()]
        agg = _aggregate_state(states)
        done = sum(s == RunState.DONE for s in states)
        print(f'{name:16} {_GLYPH.get(agg, "?")} {agg:9} {done}/{len(states)} tasks')


def cmd_status(args: argparse.Namespace) -> None:
    recs = _store_for(args.name, args).records()
    if not recs:
        raise SystemExit(f'no tasks found for experiment {args.name!r}')
    state = _aggregate_state([_rec_state(r) for r in recs])
    print(f'{args.name}  —  {state}  ({len(recs)} tasks)')
    for rec in recs:
        print(_memo_line(rec))


def cmd_results(args: argparse.Namespace) -> None:
    store = _store_for(args.name, args)
    recs = store.records()
    if not recs:
        raise SystemExit(f'no tasks found for experiment {args.name!r}')
    for rec in recs:
        key = rec['key']
        if _rec_state(rec) == RunState.DONE:
            print(f'{key}  {store.result(key)}')
        else:
            print(f'{key}  ({_rec_state(rec)} — no result)')


def cmd_logs(args: argparse.Namespace) -> None:
    print(_store_for(args.name, args).error(args.key))


def cmd_cancel(args: argparse.Namespace) -> None:
    apparatus = _build_apparatus(args.name, args)
    cancelled = apparatus.cancel(apparatus.memo_store())
    if cancelled:
        print(f'cancelled {len(cancelled)} task(s): {", ".join(cancelled)}')
    else:
        print('nothing to cancel (no in-flight tasks)')


def main() -> None:
    parser = argparse.ArgumentParser(prog='mini', description='Run and monitor memoized mi-ni experiments.')
    sub = parser.add_subparsers(dest='command', required=True)

    def _add_app_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument('--app', default='local', help='backend to read/run on: "local" or "modal"')

    p = sub.add_parser('run', help='advance a (multi-step) memoized orchestration')
    p.add_argument('path')
    p.add_argument('-w', '--watch', action='store_true', help='drive to completion with a live progress bar')
    p.add_argument('--poll', type=float, default=0.5, help='seconds between record polls while watching')
    _add_app_flag(p)
    p.add_argument('--workers', type=int, default=1, help='local worker threads / task concurrency')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('ls', help='list local experiments and their task state')
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser('status', help='show per-task state + metrics, by experiment NAME')
    p.add_argument('name')
    _add_app_flag(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser('results', help='print per-task results, by experiment NAME')
    p.add_argument('name')
    _add_app_flag(p)
    p.set_defaults(func=cmd_results)

    p = sub.add_parser('logs', help="print a task's traceback")
    p.add_argument('name')
    p.add_argument('key')
    _add_app_flag(p)
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser('cancel', help='stop in-flight tasks and mark them cancelled')
    p.add_argument('name')
    _add_app_flag(p)
    p.set_defaults(func=cmd_cancel)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

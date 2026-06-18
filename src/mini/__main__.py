"""
``python -m mini`` — drive detached experiments across short-lived processes.

Each subcommand is a quick, stateless call against the durable run state, so an
agent (or you) can launch, poll, fix, and gather without holding a session open:

    python -m mini run    experiments/pipeline.py --watch  # drive a DAG to completion
    python -m mini launch experiments/toy.py   # → toy/3f9a   (detached)
    python -m mini ls                            # experiments + their runs/tasks
    python -m mini status toy                    # latest run (or `mini run` tasks), by NAME
    python -m mini status toy/3f9a               # a specific run
    python -m mini logs   toy/3f9a --job 2       # full traceback
    python -m mini retry  toy/3f9a --failed      # re-run failed jobs
    python -m mini cancel toy/3f9a
    python -m mini results toy/3f9a

`ls`/`status` span both state models: run/job runs (`launch`/`submit`) and the
memoized orchestration tasks (`run`/`--watch`), which are addressed by name.
"""

from __future__ import annotations

import argparse
import time

from mini.apparatus import Apparatus
from mini.experiment import load_experiment
from mini.local_apparatus import LocalApparatus
from mini.memo import MemoStore
from mini.orchestration import tick
from mini.runs import DATA_ROOT, SETTLED, RunState, Run, latest_run, open_experiment, open_run

_GLYPH = {
    RunState.PENDING: '·',
    RunState.RUNNING: '▸',
    RunState.DONE: '✓',
    RunState.FAILED: '✗',
    RunState.CANCELLED: '⊘',
}


def _build_apparatus(name: str, args: argparse.Namespace) -> Apparatus:
    """Construct the apparatus the experiment runs on, from CLI flags.

    Compute is an execution choice, not part of the experiment definition. Only
    ``local`` is wired up so far; ``modal`` lands with the Modal backend.
    """
    backend = getattr(args, 'app', 'local')
    if backend == 'local':
        return LocalApparatus(name, max_workers=getattr(args, 'workers', 1))
    if backend == 'modal':
        from mini.modal_apparatus import ModalApparatus

        return ModalApparatus(name)
    raise SystemExit(f'--app {backend!r} not supported (use "local" or "modal")')


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


def _memo_store(name: str) -> MemoStore:
    """The memo store for an experiment by name (the `mini run` state model)."""
    return MemoStore(DATA_ROOT / name)


def _aggregate_state(states: list[RunState]) -> RunState:
    """Roll per-task states up to one experiment state (mirrors ``Run.state``)."""
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


def cmd_launch(args: argparse.Namespace) -> None:
    exp = load_experiment(args.path)
    print(exp.submit(_build_apparatus(exp.name, args)).id)


def cmd_ls(args: argparse.Namespace) -> None:
    root = DATA_ROOT
    names = sorted(p.name for p in root.glob('*') if (p / '.control').is_dir()) if root.exists() else []
    if not names:
        print('no experiments yet (nothing under .mini/*/.control)')
        return
    for name in names:
        recs = _memo_store(name).records()  # `mini run` orchestration tasks
        _, tokens = open_experiment(name)  # `mini launch` run/job runs
        summary = []
        if recs:
            states = [_rec_state(r) for r in recs]
            done = sum(s == RunState.DONE for s in states)
            summary.append(
                f'{_GLYPH.get(_aggregate_state(states), "?")} {_aggregate_state(states):9} {done}/{len(states)} tasks'
            )
        if tokens:
            summary.append(f'{len(tokens)} run{"s" * (len(tokens) != 1)}')
        print(f'{name:16} {"  ·  ".join(summary) or "(empty)"}')
        for token in tokens[:5]:
            run = open_run(f'{name}/{token}')
            states = [s.state for s in run.status()]
            done = sum(s == RunState.DONE for s in states)
            print(f'    {token}  {run.state():9}  {done}/{len(states)} done')


def cmd_status(args: argparse.Namespace) -> None:
    if '/' not in args.ref and (recs := _memo_store(args.ref).records()):
        # A `mini run` orchestration: addressed by name, state lives in the memo store.
        state = _aggregate_state([_rec_state(r) for r in recs])
        print(f'{args.ref}  —  {state}  ({len(recs)} tasks)')
        for rec in recs:
            print(_memo_line(rec))
        return
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
    store = apparatus.memo_store()
    print(f'{exp.name}:')
    for rec in store.records():
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
        print(f'inspect a traceback with:  python -m mini tasklog {exp.name} <key>')
        raise SystemExit(1) from e
    print(f'✓ complete: {payload}')


def cmd_tasklog(args: argparse.Namespace) -> None:
    print(MemoStore(DATA_ROOT / args.experiment).error(args.key))


def main() -> None:
    parser = argparse.ArgumentParser(prog='mini', description='Drive detached mi-ni experiments.')
    sub = parser.add_subparsers(dest='command', required=True)

    def _add_compute_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument('--app', default='local', help='compute backend: "local" or "modal"')
        p.add_argument('--workers', type=int, default=1, help='local worker threads / job concurrency')

    p = sub.add_parser('run', help='advance a (multi-step) memoized orchestration by one wake')
    p.add_argument('path')
    p.add_argument('-w', '--watch', action='store_true', help='drive to completion with a live progress bar')
    p.add_argument('--poll', type=float, default=0.5, help='seconds between record polls while watching')
    _add_compute_flags(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('launch', help='launch a single sweep detached (one-shot run/job model)')
    p.add_argument('path')
    _add_compute_flags(p)
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser('ls', help='list experiments and their runs')
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser('tasklog', help='print a memoized task traceback')
    p.add_argument('experiment')
    p.add_argument('key')
    p.set_defaults(func=cmd_tasklog)

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

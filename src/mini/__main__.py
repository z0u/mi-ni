"""
``python -m mini`` — run and monitor memoized experiments across short-lived processes.

An experiment is a ``main(ctx)`` DAG (or a single sweep, lowered to one ``ctx.map``).
Each subcommand is a quick, stateless call against the durable memo store, so an
agent (or you) can drive, poll, and gather without holding a session open:

    python -m mini run    docs/pipeline/experiment.py --watch  # drive a DAG to completion (live bar)
    python -m mini run    docs/pipeline/experiment.py          # advance one wake, then return
    python -m mini run    docs/pipeline/experiment.py --budget 2h  # auto-cancel the run past a wall-clock budget
    python -m mini retry  docs/pipeline/experiment.py          # reset FAILED/CANCELLED, then advance
    python -m mini ls                                          # experiments + task state
    python -m mini watch  pipeline                             # block until the run settles, read-only (exit 0 iff DONE)
    python -m mini status pipeline                             # per-task state + metrics, by NAME (--json for scripts)
    python -m mini results pipeline                            # per-task results
    python -m mini logs   pipeline <key>                       # a failed task's traceback
    python -m mini explain pipeline <key>                      # why this re-ran: evidence + attempt timeline
    python -m mini lineage pipeline                            # provenance: git, who/what ran it, environment, upstreams
    python -m mini cost    pipeline                            # reconcile the run's Modal cost (post-run billing)
    python -m mini cancel pipeline                             # stop in-flight tasks
    python -m mini gc     pipeline                             # plan a memo-storage sweep (--apply to delete)
    python -m mini gc     --store                              # plan an artifact-CAS sweep (--apply to delete)

State is addressed by experiment NAME (one memo store per experiment). ``--app``
picks the backend; when omitted, every verb follows the backend the experiment
was launched on (stamped in ``.mini/<name>/.app``), then ``$MINI_APP`` /
``[tool.mini] app``, then local — so after ``run --app modal``, a plain
``status`` reads the Modal control plane.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from mini.apparatus import Apparatus
from mini.experiment import Experiment, load_experiment
from mini.gc import GRACE_DEFAULT
from mini.local_apparatus import LocalApparatus
from mini.memo import META_KEY, MemoStore
from mini.orchestration import BudgetExpired, TaskFailed, retry, tick
from mini.runs import SETTLED, RunState, data_root, is_queued, stale_heartbeat
from mini.store import _project_config
from utils.time import duration

_GLYPH = {
    RunState.PENDING: "·",
    RunState.RUNNING: "▸",
    RunState.DONE: "✓",
    RunState.FAILED: "✗",
    RunState.CANCELLED: "⊘",
}


_APP_ENV = "MINI_APP"


def _resolve_app(name: str, args: argparse.Namespace) -> str:
    """The backend to act on when ``--app`` isn't passed (#47).

    Explicit flag first; then the ``.mini/<name>/.app`` marker stamped at launch
    (per-experiment ground truth — after ``run --app modal``, a plain ``status``
    just works); then the ``MINI_APP`` env var and the ``[tool.mini] app``
    pyproject key (the marker is per-checkout, so a fresh clone — CI, a scheduled
    monitor's new environment — needs one of these to be Modal-first); finally
    ``'local'``.
    """
    if app := getattr(args, "app", None):
        return app
    marker = data_root() / name / ".app"
    if marker.is_file() and (app := marker.read_text().strip()):
        return app
    return os.environ.get(_APP_ENV) or _project_config().get("app") or "local"


def _remember_app(name: str, args: argparse.Namespace) -> None:
    """Stamp the launch backend into ``.mini/<name>/.app`` so later reads follow it."""
    d = data_root() / name
    d.mkdir(parents=True, exist_ok=True)
    (d / ".app").write_text(f"{_resolve_app(name, args)}\n")


def _peek(name: str, backend: str) -> int:
    """Best-effort task count on *backend*, for the empty-read hint. Never raises
    and never creates state — Modal may not even be configured here.
    """
    try:
        if backend == "local":
            return len(MemoStore(data_root() / name).records())
        import modal

        from mini.modal_apparatus import control_dict_name

        d = modal.Dict.from_name(control_dict_name(name))  # no create_if_missing: a peek must not create
        return sum(k != META_KEY for k in d.keys())
    except Exception:
        return 0


def _known_names() -> list[str]:
    """Experiment names with a memo store under the data root (as ``ls`` lists)."""
    root = data_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.glob("*") if (p / ".control" / "memo").is_dir())


def _load_experiment_or_hint(path: str) -> Experiment:
    """``load_experiment``, but turn the name-vs-path mistake into a hint (#57).

    ``run``/``retry`` tick the DAG, so they import the experiment module and take
    a *file*; the read verbs (``status``/``results``/``cancel``/…) address the
    store by *name*. Typing a name here otherwise dies in an unhandled
    ``ImportError`` — instead, if the argument is a known experiment name, say so.
    """
    if not Path(path).is_file():
        hint = ""
        if path in _known_names():
            hint = (
                f"\n  {path!r} is an experiment name — status/results/cancel take names; "
                "run/retry take the experiment file (e.g. docs/acts/experiment.py)"
            )
        raise SystemExit(f"no experiment file at {path!r}{hint}")
    return load_experiment(path)


def _no_tasks(name: str, args: argparse.Namespace, extra: str = "") -> SystemExit:
    """The empty-read exit: name the backend we looked on, and peek at the other
    one so a wrong default points at the right flag instead of a dead end (#47).
    """
    backend = _resolve_app(name, args)
    other = "modal" if backend == "local" else "local"
    msg = f"no tasks found for experiment {name!r} on {backend}{extra}"
    if n := _peek(name, other):
        msg += f"\n  found {n} task(s) on {other} — try: --app {other}"
    return SystemExit(msg)


def _build_apparatus(name: str, args: argparse.Namespace) -> Apparatus:
    """Construct the apparatus the experiment runs on, from CLI flags.

    Compute is an execution choice, not part of the experiment definition.
    """
    backend = _resolve_app(name, args)
    if backend == "local":
        return LocalApparatus(name, max_workers=getattr(args, "workers", 1))
    if backend == "modal":
        from mini.modal_apparatus import ModalApparatus

        app = ModalApparatus(name)
        overrides = {
            k: v
            for k, v in (
                ("gpu", getattr(args, "gpu", None)),
                ("timeout", getattr(args, "timeout", None)),
                ("max_containers", getattr(args, "max_containers", None)),
            )
            if v is not None
        }
        return app.w(**overrides) if overrides else app
    raise SystemExit(
        f'unknown backend {backend!r} — use "local" or "modal" (--app / .app marker / $MINI_APP / [tool.mini] app)'
    )


def _store_for(name: str, args: argparse.Namespace) -> MemoStore:
    """The memo store for an experiment by name, on the selected backend.

    Local reads straight off disk (no apparatus needed); ``--app modal`` builds
    the apparatus so reads hit the Modal control plane (a named ``modal.Dict``).
    """
    if _resolve_app(name, args) == "local":
        return MemoStore(data_root() / name)
    return _build_apparatus(name, args).memo_store()


def _fmt_metrics(metrics: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:g}" for k, v in metrics.items())


def _age(ts: float | None) -> str:
    return f"{time.time() - ts:.0f}s ago" if ts else "—"


def _arm_budget(store: MemoStore, args: argparse.Namespace) -> None:
    """Stamp a wall-clock (cost) budget into the run's control plane.

    ``--budget 30m`` bounds the *whole* detached sweep: a forgotten or wedged run
    settles CANCELLED once the deadline passes, enforced opportunistically by any
    later ``status`` / ``watch`` / ``--watch`` poll. Passing the flag (re)arms the
    deadline relative to now — so it also re-arms a ``retry`` past an expired
    budget — while plain re-runs that advance a multi-step DAG inherit the
    existing deadline (no flag → no change).
    """
    if not (budget := getattr(args, "budget", None)):
        return
    store.set_meta(budget=budget, deadline_at=time.time() + duration(budget))


def _budget_suffix(store: MemoStore) -> str:
    """A short ``budget 30m, 12m left`` / ``budget 30m, expired`` tag for status."""
    meta = store.meta()
    if not (deadline := meta.get("deadline_at")):
        return ""
    remaining = deadline - time.time()
    when = f"{remaining:.0f}s left" if remaining > 0 else "expired"
    return f"budget {meta.get('budget', '?')}, {when}"


def _aggregate_state(states: list[RunState]) -> RunState:
    """Roll per-task states up to one experiment state."""
    if not states or all(s == RunState.DONE for s in states):
        return RunState.DONE
    if all(s in SETTLED for s in states):
        return RunState.CANCELLED if any(s == RunState.CANCELLED for s in states) else RunState.FAILED
    return RunState.RUNNING


def _rec_state(rec: dict) -> RunState:
    return RunState(rec["state"]) if rec.get("state") else RunState.PENDING


def _print_records(store: MemoStore, records: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """Print every record — current first, superseded marked — and return the split.

    A superseded record's key is one the last tick no longer requested (its fn was
    edited, its config removed). It stays visible (it may hold a result someone
    cares about, or an orphaned worker still burning), but it is *not* part of the
    run's state: aggregates and the failed-task hint consider current records only,
    so a completed run reads DONE even when an old key settled FAILED.
    """
    current, stale = store.split_current(store.records() if records is None else records)
    kept = set(store.meta().get("kept_stale") or ())  # DONE served under old code (--keep-stale-done)
    for rec in current:
        print(_memo_line(rec) + ("  (stale code — kept)" if rec["key"] in kept else ""))
    for rec in stale:
        print(f"{_memo_line(rec)}  (superseded)")
    return current, stale


def _memo_line(rec: dict) -> str:
    """One status line for a memoized task record (shared by `run`/`status`).

    A RUNNING record with no ``env`` yet reads ``queued`` — launched, but no
    worker has started (see :func:`mini.runs.is_queued`). Its ``heartbeat_at``
    is still the launch stamp, so it's shown as time-in-queue, not liveness.
    """
    state = _rec_state(rec)
    queued = is_queued(rec)
    glyph, label = ("◌", "queued") if queued else (_GLYPH.get(state, "?"), str(state))
    line = f"  {glyph} {rec.get('fn', 'task'):14} {rec['key']:26} {label:9}"
    if rec.get("total"):
        line += f"  {rec.get('step', 0)}/{rec['total']}"
    if rec.get("metrics"):
        line += f"  {_fmt_metrics(rec['metrics'])}"
    if state == RunState.RUNNING and rec.get("heartbeat_at"):
        line += f"  ⧖ queued {_age(rec['heartbeat_at'])}" if queued else f"  ♥ {_age(rec['heartbeat_at'])}"
        if stale_heartbeat(rec):
            line += "  ⚠ stale — worker may be dead"
    if gpu := rec.get("env", {}).get("gpu"):
        line += f"  on {gpu}"  # what it actually ran on, when not the local CPU
    if rec.get("fc_id"):
        line += f"  [{rec['fc_id']}]"  # Modal FunctionCall id — for log lookup / liveness
    if rec.get("error"):
        line += f"  !! {rec['error']}"
    return line


def cmd_run(args: argparse.Namespace) -> None:
    """One wake of a (possibly multi-step) orchestration: advance + report.

    With ``--watch``, instead drive the DAG to completion with a live progress
    bar; Ctrl-C stops watching (detached workers live on — re-run to resume).
    """
    exp = _load_experiment_or_hint(args.path)
    apparatus = _build_apparatus(exp.name, args)
    _remember_app(exp.name, args)
    _run(exp, apparatus, args)


def cmd_retry(args: argparse.Namespace) -> None:
    """Reset FAILED/CANCELLED tasks (or one ``--key``) then advance the DAG.

    FAILED/CANCELLED are terminal under unchanged code, so a plain ``run`` won't
    re-launch them; this is the explicit lever for a *flaky* failure. (After a
    code fix, plain ``run`` relaunches them by itself — the record's evidence is
    stale.) Fresh DONE tasks stay memo hits — to re-run one, edit its fn or bump
    ``version=``.
    """
    exp = _load_experiment_or_hint(args.path)
    apparatus = _build_apparatus(exp.name, args)
    _remember_app(exp.name, args)
    reset = retry(apparatus.memo_store(), key=args.key)
    print(f"retrying {len(reset)} task(s): {', '.join(reset) or '(none failed/cancelled)'}")
    _run(exp, apparatus, args)


def _detected_upstreams(exp: Experiment, store: MemoStore) -> dict[str, list[str]]:
    """Upstream experiments detected from the refs this run's tasks resolved.

    Each settled task records the shared refs it read and the experiment stamped
    on each at ``set_ref`` time (``upstream_refs`` — see ``mini._taskworker``).
    Rolled up here to producer → the resolved ref names (the evidence), with
    self-reads dropped (an experiment reading its own refs is not a dependency).
    """
    found: dict[str, set[str]] = {}
    for rec in store.records():
        for item in rec.get("upstream_refs") or []:
            if (producer := item.get("experiment")) and producer != exp.name:
                found.setdefault(producer, set()).add(item["ref"])
    return {name: sorted(refs) for name, refs in found.items()}


def _stamp_lineage(exp: Experiment, store: MemoStore, args: argparse.Namespace) -> None:
    """Capture run-level lineage into meta at each wake (never fails a run).

    The latest capture wins (edits re-run tasks, so the final code state is what
    produced the current results) while first-run breadcrumbs survive across wakes.
    Upstream experiments are the declared ones (``Experiment.deps``) plus the ones
    *detected* from the refs this run's tasks resolved; each is snapshotted from
    its own stored lineage, so a run records exactly which A its inputs came from.
    """
    from mini.lineage import merge_run_lineage, run_lineage, upstream_snapshot

    try:
        fresh = run_lineage()
    except Exception:  # lineage is diagnostic — never let it take a run down
        return
    detected = _detected_upstreams(exp, store)
    upstreams = []
    for dep in dict.fromkeys([*(exp.deps or []), *sorted(detected)]):
        try:
            dep_meta = _store_for(dep, args).meta()
        except Exception:
            dep_meta = {}  # an unreachable upstream still gets its minimal record below
        snap: dict[str, Any]
        if dep_meta.get("lineage"):
            snap = upstream_snapshot(dep, dep_meta)
        elif dep in detected:  # detected from a stamped ref, so the dependency is real
            snap = {"experiment": dep}
        else:  # declared but never run (or unreadable) — nothing to record
            continue
        if refs := detected.get(dep):
            snap["refs"] = refs
        upstreams.append(snap)
    if upstreams:
        fresh["upstreams"] = upstreams
    try:
        store.set_meta(lineage=merge_run_lineage(store.meta().get("lineage"), fresh))
    except Exception:
        pass


def _run(exp, apparatus: Apparatus, args: argparse.Namespace) -> None:
    """Drive one wake (or to completion with ``--watch``) and report."""
    store = apparatus.memo_store()
    _arm_budget(store, args)
    _stamp_lineage(exp, store, args)
    keep_stale = getattr(args, "keep_stale", False)
    if args.watch:
        _watch(exp, apparatus, poll=args.poll, keep_stale=keep_stale)
        # Tasks settled *during* the watch, after the stamp above — re-stamp so
        # the upstreams their records detected land in this same wake's lineage
        # (a step-per-wake run picks them up at the next wake's stamp instead).
        _stamp_lineage(exp, store, args)
        return
    if store.budget_expired():  # over budget — settle in-flight work, don't launch a new stage
        cancelled = apparatus.enforce_budget(store)
        print(f"{exp.name}:")
        _print_records(store)
        print(f"⊘ wall-clock budget elapsed — cancelled {len(cancelled)} in-flight task(s); run settled CANCELLED")
        return
    try:
        done, payload = tick(exp, apparatus, keep_stale=keep_stale)
    except ExceptionGroup, TaskFailed:  # a depended-on task settled terminally
        done, payload = False, None
    print(f"{exp.name}:")
    current, _ = _print_records(store)
    if done:
        print(f"✓ complete: {payload}")
    elif failed := [r for r in current if _rec_state(r) in (RunState.FAILED, RunState.CANCELLED)]:
        print(f"✗ {len(failed)} task(s) failed (terminal) — fix, then: python -m mini retry {args.path}")
        print(f"   see a traceback with:  python -m mini logs {exp.name} <key>")
    else:
        print(f"… suspended — {payload} (re-run to advance)")


def _watch(exp, apparatus: Apparatus, poll: float, keep_stale: bool = False) -> None:
    """Drive an orchestration to completion with a live bar (the ``--watch`` path)."""
    from mini.monitor import drive_and_watch

    try:
        payload = drive_and_watch(exp, apparatus, poll=poll, keep_stale=keep_stale)
    except KeyboardInterrupt:
        print("\n… stopped watching; tasks keep running. Re-run the same command to resume.")
        return
    except BudgetExpired as e:  # intentional teardown at the deadline, not a failure
        print(f"⊘ {e}; run settled CANCELLED")
        return
    except (ExceptionGroup, TaskFailed) as e:
        raised = e.exceptions if isinstance(e, BaseExceptionGroup) else (e,)
        failures = [tf for tf in raised if isinstance(tf, TaskFailed)]
        print(f"✗ {len(failures)} task(s) settled without completing:")
        for tf in failures:
            print(f"  ✗ {tf.key} ({tf.state})")
        print(f"inspect a traceback with:  python -m mini logs {exp.name} <key>")
        raise SystemExit(1) from e
    print(f"✓ complete: {payload}")


def cmd_ls(args: argparse.Namespace) -> None:
    names = _known_names()
    if not names:
        print("no experiments yet (run one with: python -m mini run <path>)")
        return
    root = data_root()
    for name in names:
        store = MemoStore(root / name)
        current, stale = store.split_current(store.records())
        states = [_rec_state(r) for r in current]
        agg = _aggregate_state(states)
        done = sum(s == RunState.DONE for s in states)
        line = f"{name:16} {_GLYPH.get(agg, '?')} {agg:9} {done}/{len(states)} tasks"
        if stale:
            line += f"  (+{len(stale)} superseded)"
        print(line)


def cmd_status(args: argparse.Namespace) -> None:
    apparatus = _build_apparatus(args.name, args)
    store = apparatus.memo_store()
    apparatus.reap_dead(store)  # a worker that died mid-run shouldn't read as RUNNING forever
    apparatus.enforce_budget(store)  # a forgotten over-budget run settles CANCELLED when polled
    recs = store.records()
    if not recs:
        raise _no_tasks(args.name, args)
    current, stale = store.split_current(recs)
    state = _aggregate_state([_rec_state(r) for r in current])
    if getattr(args, "json", False):
        print(json.dumps(_status_json(args.name, args, state, store, current, stale)))
        return
    header = f"{args.name}  —  {state}  ({len(current)} tasks)"
    if suffix := _budget_suffix(store):
        header += f"  ·  {suffix}"
    print(header)
    _print_records(store, recs)


def _status_json(
    name: str, args: argparse.Namespace, state: RunState, store: MemoStore, current: list[dict], stale: list[dict]
) -> dict[str, Any]:
    """The ``status --json`` payload — the agent-facing twin of the human lines.

    One JSON object on stdout, keyed for scripts (``jq -r .state``), with the
    same read-path semantics as plain ``status`` (reap + budget enforcement have
    already run). Field names are a stable contract: change additively only.
    ``state`` aggregates *current* tasks; superseded records ride along flagged,
    since an orphaned old-code worker may still be burning money.
    """
    out: dict[str, Any] = {
        "experiment": name,
        "app": _resolve_app(name, args),
        "state": str(state),
        "settled": all(_rec_state(r) in SETTLED for r in current),
        "tasks": [
            {**_task_json(rec), "superseded": superseded}
            for recs, superseded in ((current, False), (stale, True))
            for rec in recs
        ],
    }
    meta = store.meta()
    if deadline := meta.get("deadline_at"):
        out["budget"] = {
            "budget": meta.get("budget"),
            "deadline_at": deadline,
            "remaining_s": round(max(0.0, deadline - time.time()), 1),
        }
    if kept := meta.get("kept_stale"):
        out["kept_stale"] = sorted(kept)
    return out


def _task_json(rec: dict) -> dict[str, Any]:
    """One task record, trimmed to the fields the human line surfaces (plus timestamps)."""
    out: dict[str, Any] = {
        "key": rec["key"],
        "fn": rec.get("fn"),
        "state": str(_rec_state(rec)),
        "queued": is_queued(rec),
    }
    for f in ("step", "total", "metrics", "error", "exc_type", "fc_id", "env", "started_at", "finished_at"):
        if (v := rec.get(f)) is not None:
            out[f] = v
    if _rec_state(rec) == RunState.RUNNING and (hb := rec.get("heartbeat_at")):
        out["heartbeat_age_s"] = round(time.time() - hb, 1)
        out["stale_heartbeat"] = stale_heartbeat(rec)
    return out


def cmd_watch(args: argparse.Namespace) -> None:
    """Render live bars for a run by NAME until it settles — read-only (never ticks).

    The read-only twin of ``run --watch``: it renders a run this process didn't
    launch (e.g. a detached/Modal run), polling the durable records without ever
    advancing the DAG. Ctrl-C stops watching; the workers live on.
    """
    from mini.monitor import watch

    apparatus = _build_apparatus(args.name, args)
    if not apparatus.memo_store().records():
        raise _no_tasks(args.name, args, " (nothing to watch — launch it with: run)")
    try:
        records = watch(apparatus, poll=args.poll)
    except KeyboardInterrupt:
        print("\n… stopped watching; tasks keep running. Re-run to resume.")
        return
    state = _aggregate_state([_rec_state(r) for r in records])
    print(f"{args.name}  —  {state}  ({len(records)} tasks)")
    if state != RunState.DONE:  # exit code = settle outcome, so scripts can gate on it
        raise SystemExit(1)


def cmd_results(args: argparse.Namespace) -> None:
    store = _store_for(args.name, args)
    recs = store.records()
    if not recs:
        raise _no_tasks(args.name, args)
    current, stale = store.split_current(recs)
    for rec in current:
        key = rec["key"]
        if _rec_state(rec) == RunState.DONE:
            print(f"{key}  {store.result(key)}")
        else:
            print(f"{key}  ({_rec_state(rec)} — no result)")
    if stale:  # results under keys the DAG no longer requests would mislead a gather
        print(f"({len(stale)} superseded record(s) omitted — see: status)")


def cmd_logs(args: argparse.Namespace) -> None:
    print(_store_for(args.name, args).error(args.key))


def _attempt_delta(prev: dict, cur: dict) -> str:
    """Name what moved between two attempts' evidence — why *cur* re-ran."""
    bits: list[str] = []
    if prev.get("version") != cur.get("version"):
        bits.append(f"version: {prev.get('version', '-')} → {cur.get('version', '-')}")
    a, b = prev.get("deps") or {}, cur.get("deps") or {}
    for name in sorted(a.keys() | b.keys()):
        if a.get(name) == b.get(name):
            continue
        bits.append(f"{name}: {'changed' if name in a and name in b else ('added' if name in b else 'removed')}")
    return ", ".join(bits) or "retried (evidence unchanged)"


def _exec_env_summary(store: MemoStore) -> dict[str, Any]:
    """Aggregate the per-task execution environments a run actually ran on.

    Each settled task carries an ``env`` (see :func:`mini.runs.compute_env`); this
    rolls the fan-out up into the distinct GPUs / regions / clouds / hosts and the
    number of Modal containers, plus the summed execution wall time — the "what did
    this run *on*" companion to the driver-side lineage.
    """
    # env field -> the distinct values seen across tasks (each rolled up below).
    seen: dict[str, set[str]] = {k: set() for k in ("gpu", "region", "cloud", "host", "modal_task_id")}
    wall = 0.0
    for rec in store.records():
        env = rec.get("env") or {}
        if gpu := env.get("gpu"):
            env = {**env, "gpu": f"{gpu}×{env['gpu_count']}" if env.get("gpu_count") else gpu}
        for field, values in seen.items():
            if val := env.get(field):
                values.add(val)
        if (s := rec.get("started_at")) and (f := rec.get("finished_at")):
            wall += max(0.0, f - s)
    out: dict[str, Any] = {
        label: sorted(seen[field])
        for field, label in (("gpu", "gpus"), ("region", "regions"), ("cloud", "clouds"), ("host", "hosts"))
        if seen[field]
    }
    if seen["modal_task_id"]:
        out["modal_containers"] = len(seen["modal_task_id"])
    if wall:
        out["task_wall_seconds"] = round(wall, 1)
    return out


def _print_git_lineage(git: dict[str, Any], name: str) -> None:
    """Print the git block of a run's lineage (code state, remotes, dirty diff)."""
    head = git.get("describe") or git.get("short_sha") or "?"
    print(f"  code    {head}{'  · dirty' if git.get('dirty') else ''}")
    if branch := git.get("branch"):
        print(f"          branch {branch}" + (f" · tags {', '.join(git['tags'])}" if git.get("tags") else ""))
    for remote, url in (git.get("remotes") or {}).items():
        print(f"          remote {remote} {url}")
    if subject := git.get("subject"):
        print(f"          “{subject}” ({git.get('committed_at', '?')})")
    if git.get("diff"):
        trunc = " (truncated)" if git.get("diff_truncated") else ""
        print(f"          working-tree diff recorded{trunc} — see: python -m mini lineage {name} --diff")
    if untracked := git.get("untracked"):
        print(f"          untracked: {', '.join(untracked[:8])}" + (" …" if len(untracked) > 8 else ""))


def cmd_lineage(args: argparse.Namespace) -> None:
    """Print a run's captured provenance — enough to reproduce or forensically trace it.

    Shows the code state (git sha/branch/tags/remote, and whether the tree was
    dirty), who and what drove it (operator handle + AI agents), the spawning environment,
    the timeline, any upstream experiments it built on, and a rollup of what the
    tasks actually executed on. ``--diff`` dumps the recorded working-tree diff.
    """
    store = _store_for(args.name, args)
    lin = store.meta().get("lineage")
    if not lin:
        raise SystemExit(
            f"no lineage recorded for {args.name!r} — re-run it so this build can capture it "
            "(or it may have run on a backend this checkout can't read; try --app)"
        )
    git = lin.get("git") or {}
    if args.diff:
        print(git.get("diff") or "(tree was clean — no diff recorded)")
        return
    print(f"{args.name} — lineage")
    if git:
        _print_git_lineage(git, args.name)
    when = f"  when    first {lin.get('first_captured_at', '?')} · last {lin.get('captured_at', '?')}"
    print(when + (f" · {lin['wakes']} wake(s)" if lin.get("wakes") else ""))
    if ops := lin.get("operators"):
        print(f"  who     {', '.join(o.get('handle', '?') for o in ops)}")
    if agents := lin.get("agents"):
        print(f"  agents  {', '.join(_fmt_agent(a) for a in agents)}")
    drv = lin.get("driver") or {}
    dl = f"  driver  {drv.get('host', '?')} · {drv.get('platform', '?')} · py{drv.get('python', '?')}"
    if runner := drv.get("runner"):
        dl += f" · {runner.get('kind', '?')}"
    print(dl)
    _print_upstreams(lin)
    if ids := store.meta().get("modal_app_ids"):
        print(f"  modal   {len(ids)} app run(s) — cost: python -m mini cost {args.name}")
    if ran_on := _ran_on_line(store):
        print(f"  ran on  {ran_on}")


def _print_upstreams(lin: dict[str, Any]) -> None:
    """Print each upstream experiment this run built on, with its refs evidence."""
    for up in lin.get("upstreams") or []:
        code = up.get("git_describe") or (up.get("git_sha") or "?")[:12]
        line = f"  ⇐ {up.get('experiment', '?')}  {code}{'  (dirty)' if up.get('git_dirty') else ''}  {up.get('run_at', '')}"
        if refs := up.get("refs"):  # detected from these resolved refs (vs. declared in deps=)
            line += f"\n      via {', '.join(refs)}"
        print(line)


def _ran_on_line(store: MemoStore) -> str:
    """One-line rollup of what a run's tasks actually executed on (or empty)."""
    s = _exec_env_summary(store)
    bits = []
    if s.get("gpus"):
        bits.append("gpu " + ", ".join(s["gpus"]))
    if s.get("regions"):
        bits.append("region " + ", ".join(s["regions"]))
    if s.get("modal_containers"):
        bits.append(f"{s['modal_containers']} container(s)")
    if s.get("task_wall_seconds"):
        bits.append(f"{s['task_wall_seconds']:g}s task wall")
    return " · ".join(bits)


def _fmt_agent(a: dict[str, str]) -> str:
    s = a.get("name", "?")
    if v := a.get("version"):
        s += f" {v}"
    extra = [f"{k}={a[k]}" for k in ("entrypoint", "model") if a.get(k)]
    return s + (f" ({', '.join(extra)})" if extra else "")


def cmd_cost(args: argparse.Namespace) -> None:
    """Reconcile a run's Modal cost from the billing API (post-run; billing lags).

    Sums the cost of every Modal app instance this run launched (recorded in meta at
    spawn), with a per-resource breakdown (CPU / Memory / each GPU type). Only Modal
    runs have a cost; billing is at daily resolution and lags the run, so a
    just-finished run may report nothing yet.
    """
    store = _store_for(args.name, args)
    meta = store.meta()
    ids = meta.get("modal_app_ids") or []
    if not ids:
        raise SystemExit(f"{args.name!r}: no Modal app runs recorded — cost is available for Modal runs only")
    from mini.modal_apparatus import query_cost

    since = (meta.get("lineage") or {}).get("first_captured_at_epoch")
    try:
        rep = query_cost(ids, since_epoch=since)
    except Exception as e:  # billing API is a network call; a clean message beats a traceback
        raise SystemExit(f"could not query Modal billing for {args.name!r}: {e}") from e
    print(f"{args.name} — Modal cost  ${rep['total']:.4f}  ({len(ids)} app run(s), {rep['intervals']} interval(s))")
    for res, cost in sorted(rep["by_resource"].items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {res:10} ${cost:.4f}")
    if rep["intervals"] == 0:
        print("  (no billing data yet — Modal bills daily and lags the run; check back later)")


def cmd_explain(args: argparse.Namespace) -> None:
    """Show a task's identity evidence and its attempt timeline.

    A record's key is *identity* (fn + inputs); each launch stamps the evidence it
    ran under (code hash, ``version=``, a short hash per tracked dependency), and
    prior attempts stay compacted on the record. ``explain`` prints the current
    evidence and walks the timeline, answering "why did this re-run" down to the
    dependency that moved between attempts.
    """
    store = _store_for(args.name, args)
    rec = store.record(args.key)
    if not rec.get("state") and not rec.get("deps"):
        raise SystemExit(f"no record for key {args.key!r} in experiment {args.name!r}")
    requested = set(store.requested_keys() or [])
    suffix = "" if not requested or args.key in requested else "  (superseded)"
    print(f"{rec['key']}  {rec.get('fn', 'task')}  {_rec_state(rec)}{suffix}")
    print(f"  code {rec.get('code_fp', '?')} · inputs {rec.get('input_fp', '?')} · version {rec.get('version', '-')}")
    deps: dict[str, str] = rec.get("deps") or {}
    for name, h in sorted(deps.items()):
        print(f"    {name:40} {h}")
    if not deps:
        print("    (no dependency manifest — record predates `explain`)")
        return
    attempts = [*(rec.get("history") or ()), rec]
    if len(attempts) < 2:
        return
    print(f"  attempts ({len(attempts)}):")
    for i, att in enumerate(attempts, 1):
        state = att.get("state") or "?"
        line = f"    #{i} {state:9}  code {att.get('code_fp', '?')}"
        if att.get("error"):
            line += f"  !! {att['error']}"
        if i > 1:
            line += f"  ⇐ {_attempt_delta(attempts[i - 2], att)}"
        print(line)


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


_GC_LABEL = {
    "superseded": "superseded record(s)",
    "attempt-files": "task(s) with stale attempt files",
    "orphan-dir": "orphaned result dir(s)",
    "staged-call": "staged call(s) for tasks no longer running",
}


def cmd_gc(args: argparse.Namespace) -> None:
    """Reclaim storage no current read path can reach. Dry run unless ``--apply``.

    Two scopes: ``mini gc <name>`` sweeps one experiment's memo state
    (superseded records with their result dirs, replaced attempt files,
    orphaned result dirs, staged calls) on whichever backend it ran on;
    ``mini gc --store`` mark-and-sweeps the project artifact CAS. Neither
    touches a current record — a DONE result is a future memo hit, and
    deleting a FAILED record would silently turn a terminal failure into a
    relaunch.
    """
    if args.store and args.name:
        raise SystemExit("pass an experiment name or --store, not both (the store sweep is project-wide)")
    if args.store:
        return _gc_store(args)
    if not args.name:
        raise SystemExit("pass an experiment name (memo sweep) or --store (artifact CAS sweep)")
    from mini.gc import apply_gc, plan_gc

    apparatus = _build_apparatus(args.name, args)
    store = apparatus.memo_store()
    io = apparatus.gc_io(store)
    recs = store.records()
    if not recs and not io.memo_tree():
        raise _no_tasks(args.name, args)
    apparatus.reap_dead(store, recs)  # a vanished worker's RUNNING record must not read as alive
    plan = plan_gc(store, recs, io)

    print(f"{args.name} — gc plan:")
    for kind, label in _GC_LABEL.items():
        if not (items := plan.by_kind(kind)):
            continue
        keys = ", ".join(i.key for i in items[:6]) + (f", +{len(items) - 6} more" if len(items) > 6 else "")
        print(f"  {label}: {len(items)}  ({_human_size(sum(i.size for i in items))})  — {keys}")
    for reason in plan.kept:
        print(f"  kept: {reason}")
    if not plan.items:
        print("  nothing to collect")
        return
    if args.apply:
        apply_gc(store, plan, io)
        print(f"reclaimed {_human_size(plan.size)}")
    else:
        print(f"dry run — pass --apply to reclaim {_human_size(plan.size)}")


def _gc_store(args: argparse.Namespace) -> None:
    """Mark-and-sweep the project artifact store (CAS). Dry run unless ``--apply``.

    Fails closed: any in-flight task, unreadable result, or unreachable
    backend aborts the sweep with nothing deleted. Unreferenced blobs younger
    than ``--grace`` are kept — the window that protects writers this checkout
    can't see (an unpushed colleague's records, a ``put`` that skipped its
    upload just before the sweep).
    """
    from mini.gc import StoreGcError, apply_store_gc, collect_store_roots, plan_store_gc
    from mini.local_apparatus import LocalApparatus
    from mini.store import _hf_token, store_bucket, store_for

    if store_bucket() and not _hf_token():
        raise SystemExit(
            f"store-bucket {store_bucket()!r} is configured but no Hugging Face token was found — "
            "sweeping the local fallback store instead of the real CAS would be misleading. "
            "Run ./go auth (or set HF_TOKEN) first."
        )
    store = store_for(data_root() / "store")
    # Settle vanished local workers first, so a crashed run's RUNNING record
    # doesn't block the sweep forever. Modal records are reaped by their own
    # verbs (status/watch); a stale one here aborts with a pointer to those.
    for name in sorted(p.name for p in data_root().glob("*") if (p / ".control" / "memo").is_dir()):
        memo = MemoStore(data_root() / name)
        LocalApparatus(name).reap_dead(memo)
    try:
        roots, notes = collect_store_roots()
    except StoreGcError as e:
        raise SystemExit(f"store gc aborted (nothing deleted): {e}") from e
    try:
        plan = plan_store_gc(store, roots, grace=duration(args.grace))
    except NotImplementedError as e:
        raise SystemExit(f"the configured store ({type(store).__name__}) does not support gc") from e

    print(f"artifact store ({type(store).__name__}) — gc plan:")
    print(f"  {plan.total_blobs} blob(s), {_human_size(plan.total_size)} total; {plan.roots} reachable sha(s)")
    print(f"  referenced: {plan.referenced}")
    for note in notes + plan.notes:
        print(f"  kept: {note}")
    if not plan.unreferenced:
        print("  nothing to collect")
        return
    sample = ", ".join(b.sha256[:12] for b in plan.unreferenced[:6])
    more = f", +{len(plan.unreferenced) - 6} more" if len(plan.unreferenced) > 6 else ""
    print(f"  unreferenced: {len(plan.unreferenced)}  ({_human_size(plan.size)})  — {sample}{more}")
    if args.apply:
        apply_store_gc(store, plan)
        print(f"reclaimed {_human_size(plan.size)}")
    else:
        print(f"dry run — pass --apply to reclaim {_human_size(plan.size)}")


def cmd_cancel(args: argparse.Namespace) -> None:
    apparatus = _build_apparatus(args.name, args)
    cancelled = apparatus.cancel(apparatus.memo_store())
    if cancelled:
        print(f"cancelled {len(cancelled)} task(s): {', '.join(cancelled)}")
    else:
        print("nothing to cancel (no in-flight tasks)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mini", description="Run and monitor memoized mi-ni experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_app_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--app",
            default=None,
            help='backend to read/run on: "local" or "modal" (default: the backend the '
            "experiment last launched on, else $MINI_APP, else [tool.mini] app, else local)",
        )

    def _add_run_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "path",
            help="the experiment file to tick, e.g. docs/acts/experiment.py "
            "(run/retry take a file; status/results/cancel take a NAME)",
        )
        p.add_argument("-w", "--watch", action="store_true", help="drive to completion with a live progress bar")
        p.add_argument("--poll", type=float, default=0.5, help="seconds between record polls while watching")
        _add_app_flag(p)
        p.add_argument("--workers", type=int, default=1, help="local worker threads / task concurrency")
        p.add_argument("--gpu", default=None, help="Modal GPU type, e.g. L4, A100 (--app modal)")
        p.add_argument("--timeout", type=int, default=None, help="per-task timeout in seconds (--app modal)")
        p.add_argument(
            "--keep-stale-done",
            action="store_true",
            dest="keep_stale",
            help="bounded hotfix: serve DONE results even where the code has since changed, "
            "re-running only cells that never finished (default: a stale DONE re-runs too)",
        )
        p.add_argument(
            "--budget",
            default=None,
            help="wall-clock (cost) budget for the whole run, e.g. 30m, 2h; "
            "a forgotten/wedged detached run settles CANCELLED once it elapses",
        )
        p.add_argument(
            "--max-containers",
            type=int,
            default=None,
            dest="max_containers",
            help="cap concurrent Modal containers (--app modal; default: unbounded)",
        )

    p = sub.add_parser("run", help="advance a (multi-step) memoized orchestration")
    _add_run_flags(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("retry", help="reset FAILED/CANCELLED tasks then advance the DAG")
    _add_run_flags(p)
    p.add_argument("--key", default=None, help="retry just this task key (default: all failed/cancelled)")
    p.set_defaults(func=cmd_retry)

    p = sub.add_parser("ls", help="list local experiments and their task state")
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser("status", help="show per-task state + metrics, by experiment NAME")
    p.add_argument("name")
    p.add_argument(
        "--json",
        action="store_true",
        help="machine-readable status: one JSON object (stable field names; for scripts/agents)",
    )
    _add_app_flag(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser(
        "watch",
        help="block until a run settles, rendering live bars — read-only (never ticks); "
        "exits 0 iff it settled DONE, so it doubles as a wake trigger for scripts",
    )
    p.add_argument("name")
    p.add_argument("--poll", type=float, default=0.5, help="seconds between record polls while watching")
    _add_app_flag(p)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("results", help="print per-task results, by experiment NAME")
    p.add_argument("name")
    _add_app_flag(p)
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("logs", help="print a task's traceback")
    p.add_argument("name")
    p.add_argument("key")
    _add_app_flag(p)
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("explain", help="show a task's identity evidence and attempt timeline (why did this re-run)")
    p.add_argument("name")
    p.add_argument("key")
    _add_app_flag(p)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("lineage", help="show a run's provenance (git, who/what ran it, environment, upstreams)")
    p.add_argument("name")
    p.add_argument("--diff", action="store_true", help="print the recorded working-tree diff instead of the summary")
    _add_app_flag(p)
    p.set_defaults(func=cmd_lineage)

    p = sub.add_parser("cost", help="reconcile a run's Modal cost from the billing API (post-run)")
    p.add_argument("name")
    _add_app_flag(p)
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser("cancel", help="stop in-flight tasks and mark them cancelled")
    p.add_argument("name")
    _add_app_flag(p)
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser(
        "gc",
        help="reclaim stale storage — an experiment's memo state by name, or --store for the artifact CAS "
        "(dry run by default)",
    )
    p.add_argument("name", nargs="?", help="experiment to sweep (omit with --store)")
    p.add_argument("--store", action="store_true", help="mark-and-sweep the project artifact store instead")
    p.add_argument(
        "--grace",
        default=GRACE_DEFAULT,
        help=f"keep unreferenced blobs younger than this (store sweep; default {GRACE_DEFAULT})",
    )
    p.add_argument("--apply", action="store_true", help="actually delete (default: print the plan only)")
    _add_app_flag(p)
    p.set_defaults(func=cmd_gc)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

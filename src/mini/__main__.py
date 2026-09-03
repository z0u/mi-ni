"""
``python -m mini`` — run and monitor memoized experiments across short-lived processes.

An experiment is a ``main(ctx)`` DAG (or a single sweep, lowered to one ``ctx.map``). Each subcommand is a quick, stateless call against the durable memo store, so an agent (or you) can drive, poll, and gather without holding a session open:

    python -m mini run    docs/pipeline/experiment.py --watch  # drive a DAG to completion (live bar) python -m mini run    docs/pipeline/experiment.py          # advance one wake, then return python -m mini run    docs/pipeline/experiment.py --budget 2h  # auto-cancel the run past a wall-clock budget python -m mini retry  docs/pipeline/experiment.py          # reset FAILED/CANCELLED, then advance python -m mini ls                                          # experiments + task state python -m mini watch  pipeline                             # block until the run settles, read-only (exit 0 iff DONE) python -m mini watch  pipeline --timeout 10m --json        # bounded wait: exit 0 done / 1 failed / 3 attention / 124 timeout python -m mini status pipeline                             # per-task state + metrics, by NAME (--json for scripts) python -m mini status pipeline --brief                     # aggregate + counts + only tasks needing attention python -m mini results pipeline                            # per-task results python -m mini logs   pipeline <key>                       # a failed task's traceback python -m mini explain pipeline <key>                      # why this re-ran: evidence + attempt timeline python -m mini lineage pipeline                            # provenance: git, who/what ran it, environment, upstreams python -m mini cost    pipeline                            # reconcile the run's Modal cost (post-run billing) python -m mini cancel pipeline                             # stop in-flight tasks python -m mini gc     pipeline                             # plan a memo-storage sweep (--apply to delete) python -m mini gc     --store                              # plan an artifact-CAS sweep (--apply to delete)

State is addressed by experiment NAME (one memo store per experiment). ``--app`` picks the backend; when omitted, every verb follows the backend the experiment was launched on (stamped in ``.mini/<name>/.app``), then ``$MINI_APP`` / ``[tool.mini] app``, then local — so after ``run --app modal``, a plain ``status`` reads the Modal control plane.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from statistics import median
from typing import Any

from mini.apparatus import Apparatus
from mini.experiment import Experiment, load_experiment
from mini.gc import GRACE_DEFAULT
from mini.local_apparatus import LocalApparatus
from mini.memo import META_KEY, MemoStore
from mini.orchestration import BudgetExpired, TaskFailed, retry, tick
from mini.runs import (
    SETTLED,
    STALE_HEARTBEAT_S,
    RunState,
    data_root,
    describe_numerics_drift,
    in_declared_phase,
    is_queued,
    merged_numerics_drift,
    numerics_drift,
    progress_age,
    stale_heartbeat,
    stale_progress,
)
from mini.store import Artifact, _project_config
from utils.time import duration

_GLYPH = {
    RunState.PENDING: "·",
    RunState.RUNNING: "▸",
    RunState.DONE: "✓",
    RunState.FAILED: "✗",
    RunState.CANCELLED: "⊘",
}


_APP_ENV = "MINI_APP"

# How to spell this CLI in hints, so they're copy-pasteable as printed. The
# `bin/mini` wrapper exports MINI_PROG=$0 (it works from any cwd, no venv);
# otherwise trust argv[0] — the console script when on PATH, else `python -m`.
PROG = os.environ.get("MINI_PROG") or ("mini" if Path(sys.argv[0]).name == "mini" else "python -m mini")


def _app_suffix(args: argparse.Namespace) -> str:
    """Echo an explicit ``--app`` into a suggested command, so the hint works verbatim."""
    return f" --app {app}" if (app := getattr(args, "app", None)) else ""


def _resolve_app(name: str, args: argparse.Namespace) -> str:
    """The backend to act on when ``--app`` isn't passed (#47).

    Explicit flag first; then the ``.mini/<name>/.app`` marker stamped at launch (per-experiment ground truth — after ``run --app modal``, a plain ``status`` just works); then the ``MINI_APP`` env var and the ``[tool.mini] app`` pyproject key (the marker is per-checkout, so a fresh clone — CI, a scheduled monitor's new environment — needs one of these to be Modal-first); finally ``'local'``.
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
    """Best-effort task count on *backend*, for the empty-read hint. Never raises and never creates state — Modal may not even be configured here."""
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

    ``run``/``retry`` tick the DAG, so they import the experiment module and take a *file*; the read verbs (``status``/``results``/``cancel``/…) address the store by *name*. Typing a name here otherwise dies in an unhandled ``ImportError`` — instead, if the argument is a known experiment name, say so.
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
    """The empty-read exit: name the backend we looked on, and peek at the other one so a wrong default points at the right flag instead of a dead end (#47)."""
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
    common = {
        k: v
        for k, v in (
            ("watchdog", getattr(args, "watchdog", None)),
            ("watchdog_grace", getattr(args, "watchdog_grace", None)),
        )
        if v is not None
    }
    # Project-wide worker environment, on both backends: a setting that decides what
    # a task computes (`XLA_FLAGS`) belongs with the project, not restated per
    # experiment, and has to be in place before the worker starts. A role's own
    # `env=` merges over this key by key (roles are applied on top).
    if env := _project_config().get("env"):
        common["env"] = env
    if backend == "local":
        app: Apparatus = LocalApparatus(name, max_workers=getattr(args, "workers", 1))
        return app.w(**common) if common else app
    if backend == "modal":
        from mini.modal_apparatus import ModalApparatus

        app = ModalApparatus(name)
        overrides = {
            k: v
            for k, v in (
                ("gpu", getattr(args, "gpu", None)),
                ("timeout", getattr(args, "timeout", None)),
                ("max_containers", getattr(args, "max_containers", None)),
                # A project-wide default, since placement is usually a property of
                # where the run's *storage* lives rather than of one experiment. A
                # role's own region= still wins (roles are applied over this).
                ("region", getattr(args, "region", None) or _project_config().get("region")),
            )
            if v is not None
        } | common
        return app.w(**overrides) if overrides else app
    raise SystemExit(
        f'unknown backend {backend!r} — use "local" or "modal" (--app / .app marker / $MINI_APP / [tool.mini] app)'
    )


def _store_for(name: str, args: argparse.Namespace) -> MemoStore:
    """The memo store for an experiment by name, on the selected backend.

    Local reads straight off disk (no apparatus needed); ``--app modal`` builds the apparatus so reads hit the Modal control plane (a named ``modal.Dict``).
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

    ``--budget 30m`` bounds the *whole* detached sweep: a forgotten or wedged run settles CANCELLED once the deadline passes, enforced opportunistically by any later ``status`` / ``watch`` / ``--watch`` poll. Passing the flag (re)arms the deadline relative to now — so it also re-arms a ``retry`` past an expired budget — while plain re-runs that advance a multi-step DAG inherit the existing deadline (no flag → no change).
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


def _rearm_hint(store: MemoStore, path: str, args: argparse.Namespace) -> str:
    """How to get a run moving again past an expired budget, as a runnable line.

    Worth spelling out because the symptom is *silence*: past the deadline a plain ``run`` launches nothing and settles what's in flight, so a run with genuinely stale work reads exactly like one with nothing left to do.
    """
    return f"  nothing launches past an expired budget — re-arm with: {PROG} run {path} --budget {store.meta().get('budget') or '30m'}{_app_suffix(args)}"


def _dag_note(store: MemoStore, state: RunState) -> str:
    """The caveat on an all-DONE run whose ``main`` hasn't reached its end yet.

    Every launched task can be DONE while the DAG still has stages to go — the last tick suspended at a ``Pending`` rather than returning. Both read "done" from the records alone, and mistaking one for the other is how a run sits finished-looking with a publish step never executed.
    """
    if state != RunState.DONE or store.dag_complete() is not False:
        return ""
    if store.budget_expired():  # the compounding trap: work left, and nothing will launch
        return "DAG suspended, and the budget has expired — re-run with --budget to advance"
    return "DAG suspended at the last tick — re-run to advance"


def _numerics_moved(records: list[dict]) -> tuple[int, dict[str, tuple[tuple[str, ...], str]]]:
    """How many DONE results were computed under numerics packages that have since moved, and what moved.

    Read-only and self-contained: each record carries the versions its worker ran under (:func:`mini.runs.compute_env`), so a view can answer this from the store alone — no import, no tick, no fingerprinting. Same comparison the driver makes on a memo hit; here it stays visible after the run, which is when someone is usually asking where a number came from. A package's recorded side is every version seen across the records — a sweep can straddle an upgrade — merged by :func:`mini.runs.merged_numerics_drift`.
    """
    drifts = [
        drift for rec in records if _rec_state(rec) == RunState.DONE and (drift := numerics_drift(rec.get("env")))
    ]
    return len(drifts), merged_numerics_drift(drifts)


def _numerics_note(records: list[dict]) -> str:
    """One line naming the numerics packages that moved under a run's results (or empty)."""
    drifted, moved = _numerics_moved(records)
    if not moved:
        return ""
    return f"⚠ {drifted} result(s) computed under different numerics: {describe_numerics_drift(moved)}"


def _numerics_drift_json(records: list[dict]) -> dict[str, Any] | None:
    """The JSON twin of :func:`_numerics_note` — same shape in the full and ``--brief`` payloads, or ``None`` when nothing moved. ``recorded`` is a list: a run that straddles an upgrade has several baselines, and a monitor should see all of them."""
    drifted, moved = _numerics_moved(records)
    if not moved:
        return None
    return {
        "tasks": drifted,
        "packages": {n: {"recorded": list(w), "current": c} for n, (w, c) in sorted(moved.items())},
    }


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

    A superseded record's key is one the last tick no longer requested (its fn was edited, its config removed). It stays visible (it may hold a result someone cares about, or an orphaned worker still burning), but it is *not* part of the run's state: aggregates and the failed-task hint consider current records only, so a completed run reads DONE even when an old key settled FAILED.
    """
    current, stale = store.split_current(store.records() if records is None else records)
    kept = set(store.meta().get("kept_stale") or ())  # DONE served under old code (--keep-stale-done)
    rates = _fleet_rates(current)
    for rec in sorted(current, key=_launch_order):
        print(_memo_line(rec, rates) + ("  (stale code — kept)" if rec["key"] in kept else ""))
    for rec in sorted(stale, key=_launch_order):
        print(f"{_memo_line(rec)}  (superseded)")
    return current, stale


def _launch_order(rec: dict) -> tuple:
    """Sort records for display: launch order, so a sweep's stages read as stages (store order is hash order — a jumble for a human scanning 50 lines)."""
    return (rec.get("started_at") or float("inf"), rec.get("fn") or "", rec["key"])


def _worry(rec: dict, rates: dict[str, float]) -> str:
    """The badge a RUNNING task's line carries, worst worry first (or nothing)."""
    if stale_heartbeat(rec):
        return "  ⚠ stale — worker may be dead"
    if stale_progress(rec):  # emitting but not advancing: the wedge signature, not a dead worker
        return f"  ⚠ no step progress for {progress_age(rec):.0f}s — worker may be wedged"
    if deviation := _metric_trouble(rec) or _slow_outlier(rec, rates) or _timeout_projection(rec):
        return f"  ⚠ {deviation}"
    return ""


def _memo_line(rec: dict, rates: dict[str, float] | None = None) -> str:
    """One status line for a memoized task record (shared by `run`/`status`).

    A RUNNING record with no ``env`` yet reads ``queued`` — launched, but no worker has started (see :func:`mini.runs.is_queued`). Its ``heartbeat_at`` is still the launch stamp, so it's shown as time-in-queue, not liveness. *rates* (sibling throughput medians) enables the deviation flags — a cell running far behind its siblings, or projected to overrun its timeout.
    """
    state = _rec_state(rec)
    queued = is_queued(rec)
    glyph, label = ("◌", "queued") if queued else (_GLYPH.get(state, "?"), str(state))
    line = f"  {glyph} {rec.get('fn', 'task'):14} {rec['key']:26} {label:9}"
    if rec.get("total"):
        line += f"  {rec.get('step', 0)}/{rec['total']}"
    # Name the declared span, so a frozen step reads as the work rather than as a
    # freeze. Without it the line shows 3300/3300 and nothing else for the length of
    # a checkpoint upload, which is the shape a wedge has too.
    if state == RunState.RUNNING and in_declared_phase(rec) and (phase := rec.get("phase")):
        line += f"  ⋯ {phase}"
    if (rate := rec.get("steps_per_min")) is not None and state == RunState.RUNNING:
        line += f"  ~{rate:g}/min"
    if rec.get("metrics"):
        line += f"  {_fmt_metrics(rec['metrics'])}"
    if state == RunState.RUNNING and rec.get("heartbeat_at"):
        line += f"  ⧖ queued {_age(rec['heartbeat_at'])}" if queued else f"  ♥ {_age(rec['heartbeat_at'])}"
        line += _worry(rec, rates or {})
    if gpu := rec.get("env", {}).get("gpu"):
        line += f"  on {gpu}"  # what it actually ran on, when not the local CPU
    if rec.get("fc_id"):
        line += f"  [{rec['fc_id']}]"  # Modal FunctionCall id — for log lookup / liveness
    if rec.get("error"):
        line += f"  !! {rec['error']}"
    return line


def cmd_run(args: argparse.Namespace) -> None:
    """One wake of a (possibly multi-step) orchestration: advance + report.

    With ``--watch``, instead drive the DAG to completion with a live progress bar; Ctrl-C stops watching (detached workers live on — re-run to resume).
    """
    exp = _load_experiment_or_hint(args.path)
    apparatus = _build_apparatus(exp.name, args)
    _remember_app(exp.name, args)
    _run(exp, apparatus, args)


def cmd_retry(args: argparse.Namespace) -> None:
    """Reset FAILED/CANCELLED tasks (or one ``--key``) then advance the DAG.

    FAILED/CANCELLED are terminal under unchanged code, so a plain ``run`` won't re-launch them; this is the explicit lever for a *flaky* failure. (After a code fix, plain ``run`` relaunches them by itself — the record's evidence is stale.) Fresh DONE tasks stay memo hits — to re-run one, edit its fn or bump ``version=``.
    """
    exp = _load_experiment_or_hint(args.path)
    apparatus = _build_apparatus(exp.name, args)
    _remember_app(exp.name, args)
    reset = retry(apparatus.memo_store(), key=args.key)
    print(f"retrying {len(reset)} task(s): {', '.join(reset) or '(none failed/cancelled)'}")
    _run(exp, apparatus, args)


def _detected_upstreams(exp: Experiment, store: MemoStore) -> dict[str, list[str]]:
    """Upstream experiments detected from the refs this run's tasks resolved.

    Each settled task records the shared refs it read and the experiment stamped on each at ``set_ref`` time (``upstream_refs`` — see ``mini._taskworker``). Rolled up here to producer → the resolved ref names (the evidence), with self-reads dropped (an experiment reading its own refs is not a dependency).
    """
    found: dict[str, set[str]] = {}
    for rec in store.records():
        for item in rec.get("upstream_refs") or []:
            if (producer := item.get("experiment")) and producer != exp.name:
                found.setdefault(producer, set()).add(item["ref"])
    return {name: sorted(refs) for name, refs in found.items()}


def _stamp_lineage(exp: Experiment, store: MemoStore, args: argparse.Namespace) -> None:
    """Capture run-level lineage into meta at each wake (never fails a run).

    The latest capture wins (edits re-run tasks, so the final code state is what produced the current results) while first-run breadcrumbs survive across wakes. Upstream experiments are the declared ones (``Experiment.deps``) plus the ones *detected* from the refs this run's tasks resolved; each is snapshotted from its own stored lineage, so a run records exactly which A its inputs came from.
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
        if cancelled:
            print(f"⊘ wall-clock budget elapsed — cancelled {len(cancelled)} in-flight task(s); run settled CANCELLED")
        else:
            print("⊘ wall-clock budget elapsed — nothing was in flight, and this wake launched nothing")
        print(_rearm_hint(store, args.path, args))
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
        print(f"✗ {len(failed)} task(s) failed (terminal) — fix, then: {PROG} retry {args.path}")
        _logs_hint(exp.name, [r["key"] for r in failed])
    else:
        print(f"… suspended — {payload} (re-run to advance)")


def _logs_hint(name: str, keys: list[str]) -> None:
    """Point at the traceback with a command that runs as printed — the first failing key filled in, not a ``<key>`` placeholder to go hunting for."""
    print(f"  see a traceback with:  {PROG} logs {name} {keys[0]}")
    if rest := keys[1:]:
        print(f"  (+{len(rest)} more failed key(s) — see: {PROG} status {name})")


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
        _logs_hint(exp.name, [tf.key for tf in failures])
        raise SystemExit(1) from e
    print(f"✓ complete: {payload}")


def cmd_ls(args: argparse.Namespace) -> None:
    names = _known_names()
    if not names:
        print(f"no experiments in this checkout yet (launch one with: {PROG} run <experiment.py>)")
        print("  (ls reads local launch state only — a run made elsewhere is reachable by name:")
        print(f"   {PROG} status <name> --app modal)")
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
        if _dag_note(store, agg):
            line += "  (DAG suspended — re-run to advance)"
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
    brief = getattr(args, "brief", False)
    if getattr(args, "json", False):
        build = _brief_json if brief else _status_json
        print(json.dumps(build(args.name, args, state, store, current, stale)))
        return
    header = f"{args.name}  —  {state}  ({len(current)} tasks{f', +{len(stale)} superseded' if stale else ''})"
    for suffix in (_budget_suffix(store), _dag_note(store, state)):
        if suffix:
            header += f"  ·  {suffix}"
    print(header)
    if note := _numerics_note(current):  # a line of its own: it qualifies every DONE result below
        print(f"  {note}")
    if brief:
        print(f"  {_fmt_counts(current)}")
        groups = _attention_groups(current)
        for g in groups[:_ATTENTION_CAP]:
            # A one-off keeps its full line (key, fc_id — the actionable detail);
            # a homogeneous mass collapses to one counted line.
            print(_memo_line(g["sample"]) if len(g["keys"]) == 1 else _grouped_attention_line(g))
        if (extra := len(groups) - _ATTENTION_CAP) > 0:
            tasks = sum(len(g["keys"]) for g in groups[_ATTENTION_CAP:])
            print(f"  … +{extra} more cause(s), {tasks} task(s) — see: {PROG} status {args.name}")
    else:
        _print_records(store, recs)


_ATTENTION_CAP = 6  # attention groups listed before an "+N more" rollup (mirrors gc's sample cap)


def _display_state(rec: dict) -> str:
    """The state a human reads: ``queued`` for a launched-but-unstarted RUNNING record (no worker yet), else the raw state — so counts and groupings line up."""
    return "queued" if is_queued(rec) else str(_rec_state(rec))


def _counts(current: list[dict]) -> dict[str, int]:
    """Task tally by state, with queued (launched, no worker yet) split out of running."""
    counts: dict[str, int] = {}
    for rec in current:
        label = _display_state(rec)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _fmt_counts(current: list[dict]) -> str:
    order = ("done", "running", "queued", "pending", "failed", "cancelled")
    counts = _counts(current)
    return " · ".join(f"{counts[label]} {label}" for label in order if counts.get(label)) or "no tasks"


# A healthy run is not the same as a run where nothing has failed, so the flags
# below also cover *deviation from expectation* — the class of trouble that used
# to need a human to notice (seen once: three of five containers running 15–30×
# slow while every liveness check read green).
_SLOW_FRACTION = 1 / 3  # of the sibling median before a cell counts as an outlier
_MIN_SIBLINGS = 3  # reporters needed before a median means anything
_WRONG_WAY_WINDOWS = 3  # consecutive minutes a metric may drift the wrong way before it's worth a look


def _fleet_rates(current: list[dict]) -> dict[str, float]:
    """Median throughput (steps/min) per task fn, over the running cells reporting one.

    A sweep's own cells are the fair yardstick for each other: same code, same shape of work, different container. Below ``_MIN_SIBLINGS`` reporters there's no fleet to compare against and the fn is left out.
    """
    rates: dict[str, list[float]] = {}
    for rec in current:
        if _rec_state(rec) == RunState.RUNNING and not is_queued(rec) and (rate := rec.get("steps_per_min")):
            rates.setdefault(rec.get("fn") or "", []).append(rate)
    return {fn: median(rs) for fn, rs in rates.items() if len(rs) >= _MIN_SIBLINGS}


def _slow_outlier(rec: dict, rates: dict[str, float]) -> str | None:
    """Is this cell crawling relative to its siblings?

    Placement, throttling, a cross-region control plane — all invisible to a liveness probe, which reads green throughout.
    """
    rate, med = rec.get("steps_per_min"), rates.get(rec.get("fn") or "")
    if not rate or not med or rate >= med * _SLOW_FRACTION:
        return None
    return f"{rate:g} steps/min — under a third of the sibling median ({med:g})"


def _timeout_projection(rec: dict, now: float | None = None) -> str | None:
    """Will this task hit its role timeout before it finishes, at its current rate?

    A timeout sized as a multiple of the expected duration is a safety net right up until throughput drops, at which point it turns into a kill switch — and the task is killed near the end, having burned the whole budget (one sweep lost a cell at step 7,895 of 7,900). Projecting it is the difference between finding out now and finding out at the timeout.
    """
    rate, total, timeout_s = rec.get("steps_per_min"), rec.get("total"), rec.get("timeout_s")
    started = rec.get("started_at")
    if not (rate and total and timeout_s and started) or _rec_state(rec) != RunState.RUNNING:
        return None
    remaining_s = max(0, total - rec.get("step", 0)) / rate * 60.0
    projected = ((now if now is not None else time.time()) - started) + remaining_s
    if projected <= timeout_s:
        return None
    return f"projected {projected / 60:.0f}m to finish, past its {timeout_s / 60:.0f}m timeout"


def _metric_trouble(rec: dict) -> str | None:
    """Have the task's own numbers gone wrong — diverged, or steadily drifting?

    Records keep the latest value of each metric plus the movement the worker measured over trailing windows, so this is a check rather than an eyeball.

    Which way is *wrong* is settled at the worker, from the job's own ``expect_metrics`` declaration — nothing here matches on metric names, because a name can't tell you whether a number is meant to climb. A metric with no declared direction still reports its movement (``metrics_delta``); it just never lands here.
    """
    if nonfinite := rec.get("metrics_nonfinite"):
        return f"{', '.join(nonfinite)} is not finite — the run has diverged"
    goals = rec.get("metric_goals") or {}
    drifting = sorted(k for k, n in (rec.get("metrics_wrong_way") or {}).items() if n >= _WRONG_WAY_WINDOWS)
    if not drifting:
        return None
    moved = ", ".join(f"{k} {'rising' if goals.get(k) == 'down' else 'falling'}" for k in drifting)
    return f"{moved} for several windows"


def _attention_cause(rec: dict, rates: dict[str, float] | None = None) -> str | None:
    """Why this task needs a look, or ``None`` if it looks healthy.

    Doubles as the grouping label, which is what collapses a homogeneous mass failure (30 identical OOMs → one line) while keeping distinct causes apart. Ordered by how much it costs to be wrong: terminal first, then liveness, then the softer deviation-from-expected signals.
    """
    if _rec_state(rec) in (RunState.FAILED, RunState.CANCELLED):
        return rec.get("error") or str(_rec_state(rec))
    if stale_heartbeat(rec):
        return "heartbeat stale — worker may be dead"
    if stale_progress(rec):
        return "no step progress — worker may be wedged"
    if is_queued(rec):  # long-queued (capacity starvation)
        hb = rec.get("heartbeat_at")
        return "queued too long" if hb and time.time() - hb > STALE_HEARTBEAT_S else None
    if _rec_state(rec) != RunState.RUNNING:
        # The deviation flags describe a task *now*, and a DONE record keeps the
        # numbers from its final window forever — so without this, a cell whose last
        # minute was slow reads "running under a third of…" after it finished, and a
        # loss that ticked up at the end parks a settled run in the attention list
        # for good. A finished run's numbers are the scientist's problem, not the
        # monitor's.
        return None
    return _metric_trouble(rec) or _slow_outlier(rec, rates or {}) or _timeout_projection(rec)


def _needs_attention(rec: dict, rates: dict[str, float] | None = None) -> bool:
    """Does a monitor need to look at this task?"""
    return _attention_cause(rec, rates) is not None


def _attention_groups(current: list[dict]) -> list[dict[str, Any]]:
    """Attention tasks grouped by ``(fn, state, cause)``, launch-ordered by first member.

    Collapsing identical failures is what keeps a wide fan-out's mass failure legible — one ``30× train_one failed`` line instead of 30 near-identical ones, the difference between noise and a diagnosis. Each group keeps its first record (``sample``) and every key, so the caller can render a one-off in full (with its key) and a mass as a counted line.
    """
    groups: dict[tuple, dict[str, Any]] = {}
    keys: dict[tuple, list[str]] = {}
    rates = _fleet_rates(current)
    for rec in sorted((r for r in current if _needs_attention(r, rates)), key=_launch_order):
        sig = (rec.get("fn"), _display_state(rec), _attention_cause(rec, rates))
        if sig not in groups:
            groups[sig] = {"fn": rec.get("fn"), "state": sig[1], "cause": sig[2], "sample": rec}
            keys[sig] = []
        keys[sig].append(rec["key"])
    return [{**g, "keys": keys[sig]} for sig, g in groups.items()]


def _grouped_attention_line(g: dict[str, Any]) -> str:
    """One collapsed line for a multi-task attention group, keeping a sample key so a human can still pull a traceback (``logs <exp> <key>``)."""
    state = g["state"]
    glyph = "◌" if state == "queued" else _GLYPH.get(RunState(state), "?")
    line = f"  {glyph} {len(g['keys'])}× {g['fn'] or 'task':13} {state:9}"
    if _rec_state(g["sample"]) in (RunState.FAILED, RunState.CANCELLED):
        if g["sample"].get("error"):
            line += f"  !! {g['cause']}"
    else:
        line += f"  ⚠ {g['cause']}"
    return f"{line}  — e.g. {g['keys'][0]}"


def _attention_json(rec: dict) -> dict[str, Any]:
    """A task entry for the ``attention`` list: the full record trimmed to what acting on it needs — key/state/error/liveness (and ``fc_id`` for Modal log lookup), without the env/metrics/timestamps bulk of the full ``--json``."""
    out = _task_json(rec)
    for f in ("env", "started_at", "finished_at", "watchdog_s", "watchdog_grace_s"):
        out.pop(f, None)  # metrics and throughput stay: they're what the deviation flags are about
    if out.pop("queued", None):
        out.pop("heartbeat_age_s", None)  # a queued record's heartbeat is just its launch stamp
        out.pop("stale_heartbeat", None)
        if hb := rec.get("heartbeat_at"):
            out["queued_s"] = round(time.time() - hb, 1)
    return out


def _budget_json(meta: dict[str, Any]) -> dict[str, Any] | None:
    if not (deadline := meta.get("deadline_at")):
        return None
    remaining = deadline - time.time()
    return {
        "budget": meta.get("budget"),
        "deadline_at": deadline,
        "remaining_s": round(max(0.0, remaining), 1),
        # Explicit, because it changes what a `run` *does*: past the deadline it
        # launches nothing at all until the budget is re-armed.
        "expired": remaining <= 0,
    }


def _brief_json(
    name: str, args: argparse.Namespace, state: RunState, store: MemoStore, current: list[dict], stale: list[dict]
) -> dict[str, Any]:
    """The compact ``--brief`` twin of :func:`_status_json` (also ``watch --json``'s summary): aggregate state, counts by state, and *only* the tasks needing attention — a sweep's healthy lines carry no signal a monitor acts on, and on a big fan-out they dominate the payload. Field names are a stable contract: change additively only."""
    out: dict[str, Any] = {
        "experiment": name,
        "app": _resolve_app(name, args),
        "state": str(state),
        "settled": all(_rec_state(r) in SETTLED for r in current),
        "counts": _counts(current),
    }
    if (complete := store.dag_complete()) is not None:
        out["dag_complete"] = complete  # settled tasks ≠ finished DAG; see `_dag_note`
    rates = _fleet_rates(current)
    attention = [r for r in sorted(current, key=_launch_order) if _needs_attention(r, rates)]
    if attention:
        # A capped list of representative task records (keys/fc_ids to act on),
        # plus a full cause→count rollup so a wide mass failure reads as its few
        # distinct causes, not N near-identical lines.
        out["attention"] = [
            {**_attention_json(r), "cause": _attention_cause(r, rates)} for r in attention[:_ATTENTION_CAP]
        ]
        out["attention_total"] = len(attention)
        out["attention_summary"] = [
            {"fn": g["fn"], "state": g["state"], "cause": g["cause"], "count": len(g["keys"])}
            for g in _attention_groups(current)
        ]
    if budget := _budget_json(store.meta()):
        out["budget"] = budget
    if stale:
        out["superseded"] = len(stale)
    if drift := _numerics_drift_json(current):  # a monitor reading only this payload would otherwise never hear it
        out["numerics_drift"] = drift
    return out


def _status_json(
    name: str, args: argparse.Namespace, state: RunState, store: MemoStore, current: list[dict], stale: list[dict]
) -> dict[str, Any]:
    """The ``status --json`` payload — the agent-facing twin of the human lines.

    One JSON object on stdout, keyed for scripts (``jq -r .state``), with the same read-path semantics as plain ``status`` (reap + budget enforcement have already run). Field names are a stable contract: change additively only. ``state`` aggregates *current* tasks; superseded records ride along flagged, since an orphaned old-code worker may still be burning money.
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
    if (complete := store.dag_complete()) is not None:
        out["dag_complete"] = complete  # settled tasks ≠ finished DAG; see `_dag_note`
    if budget := _budget_json(meta):
        out["budget"] = budget
    if kept := meta.get("kept_stale"):
        out["kept_stale"] = sorted(kept)
    if drift := _numerics_drift_json(current):
        out["numerics_drift"] = drift
    return out


def _task_json(rec: dict) -> dict[str, Any]:
    """One task record, trimmed to the fields the human line surfaces (plus timestamps)."""
    out: dict[str, Any] = {
        "key": rec["key"],
        "fn": rec.get("fn"),
        "state": str(_rec_state(rec)),
        "queued": is_queued(rec),
    }
    for f in (
        "step",
        "total",
        "metrics",
        "metrics_delta",
        "metrics_wrong_way",
        "metric_goals",
        "metrics_nonfinite",
        "steps_per_min",
        "error",
        "exc_type",
        "fc_id",
        "env",
        "started_at",
        "finished_at",
        "timeout_s",
        "watchdog_s",
        "watchdog_grace_s",
        # The declared step-free span a task is inside, if any (mini.blocking_phase).
        # Without it a frozen step and a fresh `stale_progress: false` look like a
        # contradiction; with it, "put model" says the pause is the work. `phase_at`
        # carries the same explanation across the gaps *between* spans, where a task
        # made only of transfers holds no label to show.
        "phase",
        "phase_until",
        "phase_at",
    ):
        if (v := rec.get(f)) is not None:
            out[f] = v
    if _rec_state(rec) == RunState.RUNNING and (hb := rec.get("heartbeat_at")):
        out["heartbeat_age_s"] = round(time.time() - hb, 1)
        out["stale_heartbeat"] = stale_heartbeat(rec)
    if (age := progress_age(rec)) is not None:
        # heartbeat liveness ≠ step progress: a wedged worker can keep the former
        # fresh while the latter freezes — monitors should key on this pair.
        out["progress_age_s"] = round(age, 1)
        out["stale_progress"] = stale_progress(rec)
    return out


def cmd_watch(args: argparse.Namespace) -> None:
    """Block on a run by NAME until it settles or needs attention — read-only (never ticks).

    The read-only twin of ``run --watch``: it renders a run this process didn't launch (e.g. a detached/Modal run), polling the durable records without ever advancing the DAG. Ctrl-C stops watching; the workers live on.

    The exit code names the branch, so a monitor never parses the output to decide: 0 = settled all-DONE; 1 = settled with FAILED/CANCELLED; 3 = needs attention *now* (a task settled terminally mid-stage — e.g. a watchdog fired — or a worker went stale/wedged); 124 = ``--timeout`` elapsed with work still in flight. ``--json`` swaps the live bars for one compact summary object (the ``status --brief --json`` shape plus ``outcome``/``reason``).
    """
    from mini.monitor import watch

    apparatus = _build_apparatus(args.name, args)
    store = apparatus.memo_store()
    if not store.records():
        raise _no_tasks(args.name, args, " (nothing to watch — launch it with: run)")
    as_json, timeout = getattr(args, "json", False), getattr(args, "timeout", None)
    console = None
    if as_json:  # no live bars — the one-line summary below is the whole output
        from rich.console import Console

        console = Console(quiet=True)
    try:
        current, outcome, reason = watch(
            apparatus, poll=args.poll, console=console, timeout=duration(timeout) if timeout else None
        )
    except KeyboardInterrupt:
        print("\n… stopped watching; tasks keep running. Re-run to resume.")
        return
    state = _aggregate_state([_rec_state(r) for r in current])
    if as_json:
        _, stale = store.split_current(store.records())
        payload = _brief_json(args.name, args, state, store, current, stale) | {"outcome": outcome}
        if reason:
            payload["reason"] = reason
        print(json.dumps(payload))
    else:
        if outcome == "attention":
            print(f"⚠ needs attention: {reason}")
        elif outcome == "timeout":
            print(f"⏱ {reason}")
        print(f"{args.name}  —  {state}  ({len(current)} tasks)")
        if note := _dag_note(store, state):
            # This watch never ticks, so settling is as far as it can take the run.
            print(f"  … {note} — advance it with: {PROG} run <experiment.py>{_app_suffix(args)}")
    # Exit code = the branch to take, so scripts/monitors can gate without parsing.
    code = {"attention": 3, "timeout": 124}.get(outcome, 0 if state == RunState.DONE else 1)
    if code:
        raise SystemExit(code)


# A result routinely carries a per-step metric list — a training task's `val_loss` is
# one float per step — so a sweep's raw reprs run to ~120 KB of floats, and the scalars
# worth reading are buried in them (and, for an agent, the dump lands in its context).
# The default view elides the bulk: every mapping key and every scalar prints verbatim,
# so a value shown in place of itself is the value. What is lost is length, plus the
# interior of a long numeric sequence, which prints as rounded summary statistics rather
# than a head of three floats. Each elision states how much it stands for, and `--full`
# prints the repr.
_SEQ_HEAD = 3  # elements of a sequence shown before the elision
_STR_CAP = 80  # characters of a string shown before the elision (a sha256 still fits)
_MAX_DEPTH = 6  # nesting deeper than this is elided rather than walked
_STATS_MIN = 8  # an ordered numeric sequence longer than this summarizes instead
_STATS_FMT = ".3g"  # significant figures for a summarized (and therefore rounded) number


def _elided(parts: list[str], total: int, brackets: str) -> str:
    """*parts* (already rendered) inside *brackets*, capped at :data:`_SEQ_HEAD` with a count."""
    open_c, close_c = brackets
    shown = parts[:_SEQ_HEAD]
    if total > len(shown):
        shown.append(f"… +{total - len(shown)}")
    return f"{open_c}{', '.join(shown)}{close_c}"


def _capped(text: str, *, quoted: bool) -> str:
    """*text* cut to :data:`_STR_CAP` characters, saying how many it dropped.

    *quoted* distinguishes a string value, which prints as its repr, from an already rendered one — a number's repr must not pick up a second set of quotes.
    """
    shown = text[:_STR_CAP]
    return (repr(shown) if quoted else shown) + (f"… +{len(text) - _STR_CAP} chars" if len(text) > _STR_CAP else "")


def _summarized(value: list | tuple) -> str | None:
    """A long numeric sequence as where it starts, where it ends, and how it spreads — or ``None`` if it isn't one.

    This is the bulk the abbreviated view exists for, and a head of three floats out of hundreds answers almost nothing about it: not where the run ended, not whether it spiked, not what the best epoch reached. First and last, any interior extreme, and the mean and spread answer those in less width.

    These numbers are rounded, which nothing else here is. The length floor is what contains that: a sequence short enough for the view to print in full still prints in full, and exact. Only one already losing most of itself to the elision trades the rest for a summary — so the choice is between three exact elements out of a hundred and a rounded account of all hundred, and it is the second that tells you what the run did. ``--full`` has the originals.
    """
    if len(value) <= _STATS_MIN:
        return None  # short enough that a head shows most of it, and a summary would be the longer line
    if any(isinstance(v, bool) or not isinstance(v, numbers.Real) for v in value):
        # `bool` is an `int` to Python, but a mean over elements that read as flags doesn't.
        return None
    try:
        return _stats(value)
    except ArithmeticError, ValueError, TypeError:
        # A real number that resists float arithmetic — a 10**400 int is the reachable one —
        # gets the head view instead. This whole path is a nicety, and none of it is worth
        # taking down the verb, which is the same call the bracket choice below makes.
        return None


def _stats(value: list | tuple) -> str:
    """:func:`_summarized` once *value* is known to be a long sequence of real numbers."""
    integral = all(isinstance(v, numbers.Integral) for v in value)
    fmt = ".0f" if integral else _STATS_FMT

    def num(x: float) -> str:
        return f"{x:{fmt}}"

    bits = [f"{num(value[0])} → {num(value[-1])}"]
    finite = [float(v) for v in value if math.isfinite(v)]
    if len(finite) < len(value):
        # A nan mid-curve is the thing you most want a results dump to tell you about.
        bits.append(f"{len(value) - len(finite)} non-finite")
    if finite:
        # A metric trace usually runs one way, which puts its extremes on the ends — where they
        # already print. So an extreme is named only when it is an interior one, and the absence
        # of a `min` says the sequence never went below where you can see it ending.
        ends = {float(value[0]), float(value[-1])}
        mean = math.fsum(finite) / len(finite)
        bits += [f"{k} {num(v)}" for k, v in (("min", min(finite)), ("max", max(finite))) if v not in ends]
        bits.append(f"mean {mean:{_STATS_FMT}}")
        if len(finite) > 1:
            variance = math.fsum((x - mean) ** 2 for x in finite) / (len(finite) - 1)
            bits.append(f"std {math.sqrt(variance):{_STATS_FMT}}")
    return f"{len(value)} {'ints' if integral else 'floats'}: {', '.join(bits)}"


def _abbreviated(value: Any, depth: int = 0) -> str:
    """A one-line rendering of *value* with its bulk elided — for reading, not parsing.

    Containers are walked so their shape survives: a mapping keeps every key, and a sequence keeps its first few elements plus a count of the rest — or, once it is long and numeric, :func:`_summarized` statistics over the whole of it. Dataclasses walk as their fields, which is what keeps an :class:`~mini.store.Artifact` tree from printing every child. Scalars are never rounded or reformatted — an abbreviated number that read as exact would be worse than no number at all.
    """
    if depth > _MAX_DEPTH:
        return "…"
    match value:
        case str():
            return _capped(value, quoted=True)
        case dict():
            rendered = [f"{k!r}: {_abbreviated(v, depth + 1)}" for k, v in value.items()]
            return f"{{{', '.join(rendered)}}}"  # every key, however many: the keys are the map
        case list() | tuple() | set() | frozenset():
            # By instance, not by exact type: a namedtuple is a tuple, and a KeyError
            # here would take down the verb over a cosmetic choice of bracket.
            brackets = "[]" if isinstance(value, list) else "()" if isinstance(value, tuple) else "{}"
            # Ordered sequences only: "first → last" needs an order to name, and a set has none.
            if isinstance(value, list | tuple) and (stats := _summarized(value)) is not None:
                return f"{brackets[0]}{stats}{brackets[1]}"
            head = [_abbreviated(v, depth + 1) for v in list(value)[:_SEQ_HEAD]]
            return _elided(head, len(value), brackets)
    if hasattr(value, "shape") and hasattr(value, "dtype"):  # numpy/jax array: shape, not payload
        return f"{value.dtype}{list(value.shape)}"
    if isinstance(value, Artifact):
        # Returning bulk as an artifact is the house pattern, so this is the handle a
        # listing meets most. What identifies it here is the name and the size; the
        # sha is 64 characters no `mini` verb takes as input, and a tree's children are
        # per-file blobs that would crowd out the rest of the result. `--full` has them.
        children = f", {len(value.children)} files" if value.kind == "tree" else ""
        return f"Artifact({value.name!r}, {value.size} bytes{children}, sha256={value.sha256[:12]}…)"
    if is_dataclass(value) and not isinstance(value, type):
        rendered = [f"{f.name}={_abbreviated(getattr(value, f.name), depth + 1)}" for f in fields(value)]
        return f"{type(value).__name__}({', '.join(rendered)})"
    return _capped(repr(value), quoted=False)  # anything else — scalars included — through the cap


def cmd_results(args: argparse.Namespace) -> None:
    """Print each task's result, with long sequences elided unless ``--full``."""
    store = _store_for(args.name, args)
    recs = store.records()
    if not recs:
        raise _no_tasks(args.name, args)
    current, stale = store.split_current(recs)
    if args.key:  # one task's result, without the whole sweep's dump
        matches = [r for r in [*current, *stale] if r["key"] == args.key]
        if not matches:
            raise SystemExit(f"no record for key {args.key!r} in experiment {args.name!r} — keys are listed by: status")
        current, stale = matches, []
    render = repr if args.full else _abbreviated
    for rec in sorted(current, key=_launch_order):
        key = rec["key"]
        if _rec_state(rec) == RunState.DONE:
            print(f"{key}  {render(store.result(key))}")
        else:
            print(f"{key}  ({_rec_state(rec)} — no result)")
    if stale:  # results under keys the DAG no longer requests would mislead a gather
        print(f"({len(stale)} superseded record(s) omitted — see: status)")


def cmd_logs(args: argparse.Namespace) -> None:
    store = _store_for(args.name, args)
    err = store.error(args.key)
    if err != "(no logs)":
        print(err)
        return
    rec = store.record(args.key)
    if not rec.get("state") and not rec.get("deps"):
        raise SystemExit(f"no record for key {args.key!r} in experiment {args.name!r} — keys are listed by: status")
    # A real record with nothing captured: say why, instead of a bare "(no logs)".
    raise SystemExit(f"no traceback for {args.key!r} — it is {_rec_state(rec)}; logs holds failure tracebacks only")


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

    Each settled task carries an ``env`` (see :func:`mini.runs.compute_env`); this rolls the fan-out up into the distinct GPUs / regions / clouds / hosts and the number of Modal containers, plus the summed execution wall time — the "what did this run *on*" companion to the driver-side lineage.
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
        print(f"          working-tree diff recorded{trunc} — see: {PROG} lineage {name} --diff")
    if untracked := git.get("untracked"):
        print(f"          untracked: {', '.join(untracked[:8])}" + (" …" if len(untracked) > 8 else ""))


def cmd_lineage(args: argparse.Namespace) -> None:
    """Print a run's captured provenance — enough to reproduce or forensically trace it.

    Shows the code state (git sha/branch/tags/remote, and whether the tree was dirty), who and what drove it (operator handle + AI agents), the spawning environment, the timeline, any upstream experiments it built on, and a rollup of what the tasks actually executed on. ``--diff`` dumps the recorded working-tree diff.
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
        # Echo --app so the suggestion works where this read worked (a fresh
        # checkout has no .app marker, and a bare `cost` would land on local).
        print(f"  modal   {len(ids)} app run(s) — cost: {PROG} cost {args.name}{_app_suffix(args)}")
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

    Sums the cost of every Modal app instance this run launched (recorded in meta at spawn), with a per-resource breakdown (CPU / Memory / each GPU type). Only Modal runs have a cost; billing is at daily resolution and lags the run, so a just-finished run may report nothing yet.
    """
    store = _store_for(args.name, args)
    meta = store.meta()
    ids = meta.get("modal_app_ids") or []
    if not ids:
        backend = _resolve_app(args.name, args)
        hint = ""  # same cross-backend peek as an empty status: point at the flag, not a dead end
        if backend != "modal" and (n := _peek(args.name, "modal")):
            hint = f"\n  found {n} task(s) on modal — try: --app modal"
        raise SystemExit(
            f"{args.name!r}: no Modal app runs recorded on {backend} — cost is available for Modal runs only{hint}"
        )
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

    A record's key is *identity* (fn + inputs); each launch stamps the evidence it ran under (code hash, ``version=``, a short hash per tracked dependency), and prior attempts stay compacted on the record. ``explain`` prints the current evidence and walks the timeline, answering "why did this re-run" down to the dependency that moved between attempts.
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

    Two scopes: ``mini gc <name>`` sweeps one experiment's memo state (superseded records with their result dirs, replaced attempt files, orphaned result dirs, staged calls) on whichever backend it ran on; ``mini gc --store`` mark-and-sweeps the project artifact CAS. Neither touches a current record — a DONE result is a future memo hit, and deleting a FAILED record would silently turn a terminal failure into a relaunch.
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

    Fails closed: any in-flight task, unreadable result, or unreachable backend aborts the sweep with nothing deleted. Unreferenced blobs younger than ``--grace`` are kept — the window that protects writers this checkout can't see (an unpushed colleague's records, a ``put`` that skipped its upload just before the sweep).
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
    keys = [args.key] if args.key else None
    cancelled = apparatus.cancel(apparatus.memo_store(), keys=keys)
    if cancelled:
        print(f"cancelled {len(cancelled)} task(s): {', '.join(cancelled)}")
    elif args.key:
        print(f"nothing to cancel ({args.key!r} is not in flight — keys are listed by: status)")
    else:
        print("nothing to cancel (no in-flight tasks)")


def main() -> None:
    parser = argparse.ArgumentParser(prog=PROG, description="Run and monitor memoized mi-ni experiments.")
    sub = parser.add_subparsers(dest="command")

    def _add_name_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("name", help="experiment NAME (as listed by: ls)")

    def _add_key_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("key", help="task key, e.g. train_one-bba3437c43cf (as listed by: status NAME)")

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
            "--watchdog",
            type=int,
            default=None,
            help="abort a task whose step progress stalls this many seconds (worker-side; "
            "a wedged worker settles FAILED with a stack dump instead of burning its timeout). "
            "Applies to every role unless the role sets its own watchdog=",
        )
        p.add_argument(
            "--watchdog-grace",
            type=int,
            default=None,
            dest="watchdog_grace",
            help="looser watchdog threshold until the task's first progress emission, "
            "covering one-off setup (tokenization, compilation) so --watchdog can stay "
            "tight (default: --watchdog)",
        )
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
        p.add_argument(
            "--region",
            default=None,
            help="pin Modal containers to a region, e.g. us-east (--app modal; default: "
            "[tool.mini] region, else Modal's choice — which may place a sweep's cells "
            "on different continents, far from the shared Volume/Dict). Costs a region "
            "premium; a role's own region= wins over this",
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
    _add_name_arg(p)
    p.add_argument(
        "--json",
        action="store_true",
        help="machine-readable status: one JSON object (stable field names; for scripts/agents)",
    )
    p.add_argument(
        "--brief",
        action="store_true",
        help="aggregate + counts + only the tasks needing attention (failed/stale/wedged/long-queued) "
        "— skip the per-task listing of a big sweep",
    )
    _add_app_flag(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser(
        "watch",
        help="block until a run settles or needs attention — read-only (never ticks); "
        "the exit code names the branch: 0 settled DONE, 1 settled FAILED/CANCELLED, "
        "3 attention (new failure or stale/wedged worker), 124 --timeout elapsed",
    )
    _add_name_arg(p)
    p.add_argument("--poll", type=float, default=0.5, help="seconds between record polls while watching")
    p.add_argument(
        "--timeout",
        default=None,
        help="stop watching after this long (e.g. 90s, 10m) and exit 124 if still in flight — "
        "bound the wait instead of wrapping in timeout(1)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="print one compact summary object instead of live bars (the status --brief --json "
        "shape plus outcome/reason)",
    )
    _add_app_flag(p)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("results", help="print per-task results, by experiment NAME")
    _add_name_arg(p)
    p.add_argument("key", nargs="?", default=None, help="print just this task's result (default: every task)")
    p.add_argument(
        "--full",
        action="store_true",
        help=f"print each result's whole repr; by default sequences past {_SEQ_HEAD} elements are "
        f"elided with a count, and numeric ones past {_STATS_MIN} are summarized as first/last, any "
        "interior min/max, and mean/std (a per-step metric list is otherwise thousands of floats)",
    )
    _add_app_flag(p)
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("logs", help="print a failed task's traceback")
    _add_name_arg(p)
    _add_key_arg(p)
    _add_app_flag(p)
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("explain", help="show a task's identity evidence and attempt timeline (why did this re-run)")
    _add_name_arg(p)
    _add_key_arg(p)
    _add_app_flag(p)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("lineage", help="show a run's provenance (git, who/what ran it, environment, upstreams)")
    _add_name_arg(p)
    p.add_argument("--diff", action="store_true", help="print the recorded working-tree diff instead of the summary")
    _add_app_flag(p)
    p.set_defaults(func=cmd_lineage)

    p = sub.add_parser("cost", help="reconcile a run's Modal cost from the billing API (post-run)")
    _add_name_arg(p)
    _add_app_flag(p)
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser("cancel", help="stop in-flight tasks and mark them cancelled")
    _add_name_arg(p)
    p.add_argument(
        "--key",
        default=None,
        help="cancel just this task (e.g. a wedged worker), leaving healthy siblings running "
        "(default: every in-flight task)",
    )
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
    if not args.command:
        # Bare invocation prints the synopsis alone (session-start hooks inject it
        # as a one-line map of the CLI). Exit 2 is argparse's own code for a usage
        # error — what `add_subparsers(required=True)` used to raise here.
        parser.exit(2, parser.format_usage())
    args.func(args)


if __name__ == "__main__":
    main()

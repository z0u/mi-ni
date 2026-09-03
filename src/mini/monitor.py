"""
Drive a memoized experiment to completion with a live Rich progress display.

``mini run <exp> --watch`` ticks the orchestration to launch each stage, then *polls the durable memo records* (never re-ticking to poll — see todo, "Keep `tick` (drive) distinct from polling (read)") and renders a live bar per task until the in-flight set settles, advancing the DAG stage by stage until done.

Ctrl-C only stops *watching*: the task workers are detached subprocesses, so they keep running. Re-running the same command reattaches — completed steps are memo hits and in-flight tasks aren't relaunched — so monitoring just resumes.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from mini.apparatus import Apparatus
from mini.experiment import Experiment
from mini.memo import PollCache
from mini.orchestration import BudgetExpired, tick
from mini.runs import SETTLED, RunState, in_declared_phase, is_queued, progress_age, stale_heartbeat, stale_progress

__all__ = ["drive_and_watch", "watch"]

_COLOR = {
    RunState.RUNNING: "cyan",
    RunState.DONE: "green",
    RunState.FAILED: "red",
    RunState.CANCELLED: "yellow",
}
_QUEUED_COLOR = "blue"  # launched but no worker yet (RUNNING record, no env)


def _fmt_metrics(metrics: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:g}" for k, v in metrics.items())


def _bar_desc(rec: dict[str, Any], queued: bool) -> str:
    """The label for one task's bar: key + liveness tell + message/metrics/error."""
    desc = rec["key"]
    if queued:
        desc += " — queued"
    elif stale_heartbeat(rec):
        desc += " — ♥ stale, worker may be dead"
    elif stale_progress(rec):  # heartbeat fresh but step frozen: the wedge signature
        desc += " — ⚠ no step progress, worker may be wedged"
    elif in_declared_phase(rec) and (phase := rec.get("phase")):
        desc += f" — ⋯ {phase}"  # the step is frozen because this span has no steps to report
    if rec.get("message"):
        desc += f" — {rec['message']}"
    if rec.get("metrics"):
        desc += f"  {_fmt_metrics(rec['metrics'])}"
    if rec.get("error"):
        desc += f"  !! {rec['error']}"
    return desc


def _refresh(progress: Progress, bars: dict[str, TaskID], records: list[dict[str, Any]]) -> None:
    """Reflect the latest memo records onto the live bars (one per task key)."""
    for rec in records:
        key = rec["key"]
        state = RunState(rec["state"]) if rec.get("state") else RunState.PENDING
        step, total = rec.get("step", 0), rec.get("total", 0)
        if state == RunState.DONE:  # prep steps emit no progress; show them full
            total = total or 1
            step = total
        elif state in (RunState.FAILED, RunState.CANCELLED):
            total = total or 1
        queued = is_queued(rec)  # RUNNING claimed, but no worker has started yet
        color = _QUEUED_COLOR if queued else _COLOR.get(state, "white")
        desc = f"[{color}]{escape(_bar_desc(rec, queued))}[/]"  # escape: errors/messages may hold [...]
        if key not in bars:
            bars[key] = progress.add_task(desc, total=total or None)
        progress.update(bars[key], completed=step, total=total or None, description=desc)


def _progress(console: Console | None) -> Progress:
    """The shared live-bar layout (one row per task key)."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console or Console(),
    )


def _rec_state(rec: dict[str, Any]) -> RunState:
    return RunState(rec["state"]) if rec.get("state") else RunState.PENDING


def _stale_reason(rec: dict[str, Any]) -> str | None:
    """The liveness worry on a RUNNING record, if any (both are display-only hints)."""
    if stale_heartbeat(rec):
        return "heartbeat stale — worker may be dead"
    if stale_progress(rec):  # emitting but not advancing: the wedge signature
        return f"no step progress for {progress_age(rec):.0f}s — worker may be wedged"
    return None


def watch(
    apparatus: Apparatus,
    *,
    poll: float = 0.5,
    console: Console | None = None,
    timeout: float | None = None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Watch a run this process did *not* launch, until it settles or needs a hand.

    The read-only twin of ``drive_and_watch``: it polls the durable records and reaps vanished workers, but never ``tick``s — so it never launches work. Use it to watch a detached/Modal run from another process (``mini watch <name>``); contrast ``run --watch``, which also drives the DAG forward.

    Returns ``(current_records, outcome, reason)`` where *outcome* is:

    - ``"settled"`` — every current task settled (*reason* is ``None``);
    - ``"attention"`` — something *happened* that a monitor should act on now, rather than waiting for the rest of the stage: a task settled FAILED/CANCELLED that wasn't terminal when the watch began (e.g. a watchdog fired, or a vanished worker was reaped), or a RUNNING task's liveness went stale (stale heartbeat / frozen step, held across two consecutive polls — the thresholds themselves are the debounce);
    - ``"timeout"`` — *timeout* seconds elapsed with work still in flight.

    Terminal tasks from *before* the watch never trigger attention (a run deliberately advanced past a failed cell would otherwise wake every watcher immediately). Lets ``KeyboardInterrupt`` propagate (the caller reports; workers live on).
    """
    store = apparatus.memo_store()
    cache = PollCache()  # serve the settled tail from memory; poll only what's in flight
    deadline = time.monotonic() + timeout if timeout else None
    baseline: set[str] | None = None  # keys already terminal at the first poll
    stale_polls: dict[str, int] = {}  # consecutive polls a key has looked stale
    with _progress(console) as progress:
        bars: dict[str, TaskID] = {}
        while True:
            apparatus.enforce_budget(store)  # tear down a run past its wall-clock budget (→ CANCELLED)
            records = cache.records(store)
            apparatus.reap_dead(store, records)  # settle vanished workers — read-only path, never relaunches
            _refresh(progress, bars, records)
            # Settle on the *current* records only: a superseded key (fn edited,
            # config removed) will never be requested again, so waiting on its
            # record would watch forever. Split each poll — another process may
            # tick (and re-key) while we watch.
            current, _ = store.split_current(records)
            terminal = {
                r["key"]: _rec_state(r) for r in current if _rec_state(r) in (RunState.FAILED, RunState.CANCELLED)
            }
            if baseline is None:
                baseline = set(terminal)
            if current and all(_rec_state(r) in SETTLED for r in current):
                return current, "settled", None
            if fresh := sorted(terminal.keys() - baseline):
                return current, "attention", f"{fresh[0]} settled {terminal[fresh[0]]}"
            for rec in current:
                key = rec["key"]
                stale_polls[key] = stale_polls.get(key, 0) + 1 if (why := _stale_reason(rec)) else 0
                if why and stale_polls[key] >= 2:  # held across polls — not a read racing an update
                    return current, "attention", f"{key}: {why}"
            if deadline is not None and time.monotonic() >= deadline:
                return current, "timeout", f"still in flight after {timeout:.0f}s"
            time.sleep(poll)


def drive_and_watch(
    experiment: Experiment,
    apparatus: Apparatus,
    *,
    poll: float = 0.5,
    console: Console | None = None,
    keep_stale: bool = False,
) -> Any:
    """Drive *experiment* to completion on *apparatus*, rendering a live bar.

    Returns the orchestration's payload on completion. Propagates ``TaskFailed`` (or an ``ExceptionGroup`` of them) raised by ``tick`` when a depended-on task has settled terminally — ``tick`` won't relaunch it, so re-ticking surfaces the failure rather than spinning. Lets ``KeyboardInterrupt`` propagate too (the caller reports; detached workers live on). *keep_stale* is passed through to each ``tick`` (serve DONE results whose code has since changed).
    """
    store = apparatus.memo_store()
    with _progress(console) as progress:
        bars: dict[str, TaskID] = {}
        while True:
            if store.budget_expired():  # don't launch a new stage past the deadline
                cancelled = apparatus.enforce_budget(store)
                _refresh(progress, bars, store.records())
                raise BudgetExpired(cancelled)
            done, payload = tick(experiment, apparatus, keep_stale=keep_stale)  # advance DAG, launch missing work
            # A tick can launch new keys and reset retried ones, so the cache for
            # this stage starts fresh — only the settled tail *within* a poll loop
            # is immutable, which is where the cheap re-reads pay off.
            cache = PollCache()
            _refresh(progress, bars, cache.records(store))
            if done:
                return payload
            # Read-only poll until the in-flight set settles — no re-ticking.
            while True:
                if cancelled := apparatus.enforce_budget(store):  # over budget — tear down in-flight tasks
                    _refresh(progress, bars, store.records())
                    raise BudgetExpired(cancelled)
                records = cache.records(store)
                apparatus.reap_dead(store, records)  # settle vanished workers so a kill can't wedge the drain
                _refresh(progress, bars, records)
                if not any(r.get("state") == RunState.RUNNING for r in records):
                    break
                time.sleep(poll)
            # Loop back to re-tick. Any terminal-but-not-DONE task (FAILED/CANCELLED)
            # the DAG depends on makes the re-tick raise (TaskFailed / ExceptionGroup)
            # rather than spin — tick won't relaunch it. A failed task nothing awaits
            # is harmless: the re-tick just progresses past it.

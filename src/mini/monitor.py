"""
Drive a memoized experiment to completion with a live Rich progress display.

``mini run <exp> --watch`` ticks the orchestration to launch each stage, then
*polls the durable memo records* (never re-ticking to poll — see todo, "Keep
`tick` (drive) distinct from polling (read)") and renders a live bar per task
until the in-flight set settles, advancing the DAG stage by stage until done.

Ctrl-C only stops *watching*: the task workers are detached subprocesses, so
they keep running. Re-running the same command reattaches — completed steps are
memo hits and in-flight tasks aren't relaunched — so monitoring just resumes.
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
from mini.runs import SETTLED, RunState, is_queued

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
        desc = key
        if queued:
            desc += " — queued"
        if rec.get("message"):
            desc += f" — {rec['message']}"
        if rec.get("metrics"):
            desc += f"  {_fmt_metrics(rec['metrics'])}"
        if rec.get("error"):
            desc += f"  !! {rec['error']}"
        color = _QUEUED_COLOR if queued else _COLOR.get(state, "white")
        desc = f"[{color}]{escape(desc)}[/]"  # escape: errors/messages may hold [...]
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


def watch(
    apparatus: Apparatus,
    *,
    poll: float = 0.5,
    console: Console | None = None,
) -> list[dict[str, Any]]:
    """Render a live bar for a run this process did *not* launch, until it settles.

    The read-only twin of ``drive_and_watch``: it polls the durable records and
    reaps vanished workers, but never ``tick``s — so it never launches work. Use
    it to watch a detached/Modal run from another process (``mini watch <name>``);
    contrast ``run --watch``, which also drives the DAG forward.

    Returns the final records once every task has settled. Lets
    ``KeyboardInterrupt`` propagate (the caller reports; workers live on).
    """
    store = apparatus.memo_store()
    cache = PollCache()  # serve the settled tail from memory; poll only what's in flight
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
            if current and all(_rec_state(r) in SETTLED for r in current):
                return current
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

    Returns the orchestration's payload on completion. Propagates ``TaskFailed``
    (or an ``ExceptionGroup`` of them) raised by ``tick`` when a depended-on task
    has settled terminally — ``tick`` won't relaunch it, so re-ticking surfaces the
    failure rather than spinning. Lets ``KeyboardInterrupt`` propagate too (the
    caller reports; detached workers live on). *keep_stale* is passed through to
    each ``tick`` (serve DONE results whose code has since changed).
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

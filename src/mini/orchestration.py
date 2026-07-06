"""
Memoized orchestration for multi-step experiments.

An experiment is a plain function ``main(ctx)`` that expresses the DAG in
ordinary Python. Each ``ctx.run`` / ``ctx.map`` resolves to an identity key
(fn + inputs): a record with a *current* result -> return it; otherwise launch
a detached task and *suspend* the wake by raising ``Pending``. A driver re-runs
``main`` each wake; completed steps are memo hits, so only the un-run / stale /
failed pieces execute — crash-recovery by re-run.

"Current" is judged per record against the attempt's stored evidence (code
fingerprint + ``version=``): an edit re-runs the task **in place** under the
same key. A FAILED task whose code has since changed relaunches automatically —
the fix is what it was waiting for; a DONE one re-runs too unless the tick is
told ``keep_stale`` (the bounded sweep-hotfix lever).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, overload

from mini.memo import MemoStore, task_key_parts
from mini.runs import SETTLED, RunState

if TYPE_CHECKING:
    from mini.apparatus import Apparatus
    from mini.experiment import Experiment

__all__ = ["MemoError", "Pending", "TaskFailed", "BudgetExpired", "MISSING", "Ctx", "tick", "retry"]


class MemoError(Exception):
    """Base for exceptions raised by the memoized detached path."""


class Pending(MemoError):
    """Raised to suspend the current wake until in-flight tasks finish."""


class BudgetExpired(MemoError):
    """The run blew its wall-clock (cost) budget — in-flight tasks were cancelled.

    Raised by the ``--watch`` driver when it tears a run down at its deadline, so
    the caller can report the teardown distinctly from a task *failure*: the run
    settled CANCELLED on purpose, not because anything went wrong. Carries the
    keys it cancelled (may be empty if the deadline passed between stages).
    """

    def __init__(self, cancelled: list[str]):
        self.cancelled = cancelled
        super().__init__(f"wall-clock budget elapsed — cancelled {len(cancelled)} in-flight task(s)")


class TaskFailed(MemoError):
    """A task settled FAILED/CANCELLED — terminal, so the DAG can't progress past it.

    ``ctx.run`` raises this directly; ``ctx.map`` (without ``allow_partial``) raises
    an ``ExceptionGroup`` of them — one per failed cell — so a strict fan-out
    surfaces *every* failure at once rather than the first. ``except* TaskFailed``
    handles the group ergonomically.

    This is a *report* of a past failure, not the failure itself: the task ran in a
    detached worker that has already exited, so the original exception object is
    gone. The worker's stored traceback rides along in ``.error`` (and the message),
    and ``.exc_type`` carries the original exception's fully-qualified type name (a
    plain string, so triage works even when the driver can't import the worker's
    libraries) — bucket a fan-out's failures by kind without parsing tracebacks::

        except* TaskFailed as eg:
            oom = [e for e in eg.exceptions if 'OutOfMemory' in e.exc_type]

    Recover with ``mini retry``.
    """

    def __init__(self, key: str, state: RunState, error: str = "", exc_type: str = ""):
        self.key = key
        self.state = state
        self.error = error
        self.exc_type = exc_type
        head = f"{key} settled {state}"
        super().__init__(f"{head}\n{error}" if error and error != "(no logs)" else head)


class _Missing:
    """Sentinel for a ``map`` cell that produced no result.

    ``ctx.map(..., allow_partial=True)`` returns this in the position of any task
    that settled ``FAILED``/``CANCELLED``, so results stay index-aligned with the
    inputs — downstream code commonly ``zip``s configs with results, and dropping
    cells would misalign that. It is a *falsey* singleton distinct from ``None``
    (which tasks may legitimately return), so both idioms work::

        present = [r for r in results if r]        # drop the gaps
        ok = [(c, r) for c, r in zip(cfgs, results) if r is not MISSING]
    """

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<missing>"

    def __reduce__(self) -> tuple:
        return (_Missing, ())  # round-trips to the same singleton through (cloud)pickle


MISSING = _Missing()


class Ctx:
    """The memoized, non-blocking ``run``/``map`` an orchestration calls.

    ``run`` raises ``Pending`` the moment a result isn't ready, so code after it
    only runs once the result exists. For parallel fan-out use ``map``, which
    launches *all* missing tasks before suspending.

    Each call runs on the tick's *apparatus* by default. Route a step elsewhere
    with ``role=`` (a label the experiment's ``roles`` table binds to hardware —
    the file-experiment path, since the CLI holds no apparatus handles) or ``on=``
    (a concrete apparatus — the notebook path). The apparatus also supplies the
    per-step ``before_each`` hooks.
    """

    def __init__(
        self,
        store: MemoStore,
        apparatus: Apparatus,
        roles: dict[str, Apparatus] | None = None,
        keep_stale: bool = False,
    ):
        self.store = store
        self.apparatus = apparatus
        self.roles = roles or {}
        self.keep_stale = keep_stale
        self.launched: list[str] = []
        self.pending: list[str] = []
        # Every key this wake resolved (hit, in-flight, or launched) — the DAG's
        # *requested set*. A record whose key a wake no longer requests (its fn
        # renamed, its config removed) is superseded: still on disk, but not part of
        # the run's state. ``tick`` persists this so read-only views can tell the two
        # apart without re-running ``main``.
        self.requested: list[str] = []
        # DONE results served despite stale evidence (keep_stale) — persisted so
        # read-only views can badge them (they can't fingerprint code themselves).
        self.stale_kept: list[str] = []

    def _route(self, on: Apparatus | None, role: str | None) -> Apparatus:
        """Resolve which apparatus a step runs on: ``role`` label, ``on=``, or default."""
        if on is not None and role is not None:
            raise ValueError("pass either role= or on=, not both")
        if role is not None:
            if role not in self.roles:
                known = ", ".join(sorted(self.roles)) or "(none defined)"
                raise ValueError(f"unknown role {role!r}; defined roles: {known}")
            return self.roles[role]
        return on or self.apparatus

    def _classify(
        self, fn: Callable, args: tuple, version: str | None, app: Apparatus
    ) -> tuple[str, RunState | None, tuple[str, str, Callable, tuple, list] | None]:
        """Resolve a call's key/state and, if it needs launching, claim it RUNNING
        and return its batch entry — *without* spawning. The caller batches the
        spawn so a ``map`` fans out in one ``spawn_tasks`` call.

        The key is identity; whether a settled record still *counts* is judged
        against its attempt's evidence. Stale FAILED/CANCELLED relaunches — the
        edit is the fix a terminal task was waiting for, so no ``retry`` needed.
        Stale DONE re-runs too (bias to over-invalidate) unless ``keep_stale``,
        which bounds a sweep hotfix to the cells that actually need the new code.
        Same-evidence FAILED/CANCELLED stays terminal (retry takes intent), and a
        RUNNING task is never relaunched out from under its worker — staleness
        waits for it to settle.

        Launching is a *claim*, not a blind write: ``mark_running`` replaces the
        record only if its generation still matches what we read, so two
        concurrent tickers (a cron routine and a human both waking the run)
        can't both spawn a worker for one key — the loser sees RUNNING and
        suspends like any other in-flight task.
        """
        key, parts = task_key_parts(fn, args, version)
        self.requested.append(key)
        rec = self.store.record(key)
        state = RunState(rec["state"]) if rec.get("state") else None
        stale = state is not None and (rec.get("code_fp"), rec.get("version")) != (
            parts["code_fp"],
            parts.get("version"),
        )
        keep = stale and state == RunState.DONE and self.keep_stale
        if keep:
            self.stale_kept.append(key)
        to_launch: tuple[str, str, Callable, tuple, list] | None = None
        if state is None or (stale and state != RunState.RUNNING and not keep):
            gen = self.store.mark_running(fn, key, parts, expect_gen=rec.get("gen"))
            if gen is not None:
                to_launch = (key, gen, fn, args, getattr(app, "_before_hooks", []))
                self.launched.append(key)
            state = RunState.RUNNING
        return key, state, to_launch

    def _task_failed(self, key: str, state: RunState) -> TaskFailed:
        """Build a ``TaskFailed`` for a settled-but-not-DONE key, with its stored traceback + type."""
        return TaskFailed(key, state, self.store.error(key), self.store.record(key).get("exc_type", ""))

    def run[R](
        self,
        fn: Callable[..., R],
        *args: Any,
        version: str | None = None,
        on: Apparatus | None = None,
        role: str | None = None,
    ) -> R:
        app = self._route(on, role)
        key, state, to_launch = self._classify(fn, args, version, app)
        if to_launch is not None:
            app.spawn_tasks(self.store, [to_launch])
        if state == RunState.DONE:
            return self.store.result(key)
        if state in SETTLED:  # terminal, not DONE -> FAILED/CANCELLED; surface rather than suspend
            raise self._task_failed(key, state)
        self.pending.append(key)
        raise Pending(f"waiting on {key}")

    @overload
    def map[R](
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        version: str | None = None,
        on: Apparatus | None = None,
        role: str | None = None,
        allow_partial: Literal[False] = False,
    ) -> list[R]: ...
    @overload
    def map[R](
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        version: str | None = None,
        on: Apparatus | None = None,
        role: str | None = None,
        allow_partial: Literal[True],
    ) -> list[R | _Missing]: ...
    def map[R](
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        version: str | None = None,
        on: Apparatus | None = None,
        role: str | None = None,
        allow_partial: bool = False,
    ) -> list[R] | list[R | _Missing]:
        """Fan out *fn* over the zipped *iterables*, suspending until the results
        are ready.

        Like ``Executor.map`` / ``Apparatus.map``: the iterables are zipped and
        each row is passed as positional arguments — ``ctx.map(train, lrs, sizes)``
        runs ``train(lr, size)`` per pair. With a single iterable each
        element is passed as *one* argument, never unpacked (an element that
        happens to be a tuple stays a tuple). The iterables are collected
        immediately rather than lazily. Unlike ``Executor.map``, mismatched
        iterable lengths raise rather than silently truncating the sweep.

        Launches every missing cell in one batch, then raises ``Pending`` while
        any task is still in flight. By default the map is all-or-nothing: once the
        fan-out has *settled*, any cell that settled ``FAILED``/``CANCELLED`` raises
        — all of them together, as an ``ExceptionGroup`` of ``TaskFailed`` (so you
        see every failure, not just the first). ``tick`` won't relaunch a terminal
        cell, so this is the DAG giving up rather than spinning; ``retry`` heals it.

        With ``allow_partial=True`` the map still waits for in-flight tasks, but
        once everything has settled it returns instead of raising: the result list
        stays index-aligned with the inputs, with ``MISSING`` in the position of
        each failed/cancelled cell. This lets the pipeline's later steps run on the
        subset that succeeded.
        """
        app = self._route(on, role)
        calls = list(zip(*iterables, strict=True))
        results: list[Any] = []
        batch: list[tuple[str, str, Callable, tuple, list]] = []
        failed: list[tuple[str, RunState]] = []
        # `self.pending` holds only truly in-flight keys; settled-but-failed cells go
        # to `failed`. Ctx is rebuilt each tick and the first incomplete step raises,
        # so this stays a clean per-tick view across steps.
        for args in calls:
            key, state, to_launch = self._classify(fn, args, version, app)
            if to_launch is not None:
                batch.append(to_launch)
            if state == RunState.DONE:
                results.append(self.store.result(key))
            elif state in SETTLED:  # terminal, not DONE -> FAILED/CANCELLED
                failed.append((key, state))
                results.append(MISSING)  # keep index alignment; returned only under allow_partial
            else:  # in flight
                self.pending.append(key)
                results.append(MISSING)  # placeholder; discarded — we suspend below
        if batch:  # one spawn for the whole fan-out
            app.spawn_tasks(self.store, batch)
        if self.pending:  # wait for in-flight tasks first, regardless of allow_partial
            raise Pending(f"{len(self.pending)} task(s) in flight")
        if failed and not allow_partial:  # everything settled — surface the failures together
            raise ExceptionGroup(
                f"{len(failed)} of {len(calls)} task(s) failed",
                [self._task_failed(key, state) for key, state in failed],
            )
        return results


def tick(experiment: Experiment, apparatus: Apparatus, keep_stale: bool = False) -> tuple[bool, Any]:
    """Run one wake of an experiment's orchestration on *apparatus*.

    Returns ``(done, payload)``: ``(True, result)`` if the DAG completed, or
    ``(False, reason)`` if it suspended waiting on in-flight tasks. Propagates
    ``TaskFailed`` (or an ``ExceptionGroup`` of them, from a strict ``map``) when a
    step the DAG depends on has settled terminally — the run can't progress without
    a ``retry``. Steps can override the apparatus per call via ``ctx.run(...,
    on=)`` / ``ctx.map(..., on=)``.

    *keep_stale* is the bounded-hotfix lever (``--keep-stale-done``): serve DONE
    results even when their code has since changed, so an edit re-runs only the
    cells that never finished. Stale FAILED/CANCELLED always relaunches.
    """
    store = apparatus.memo_store()
    ctx = Ctx(store, apparatus, experiment.resolve_roles(apparatus), keep_stale=keep_stale)
    complete = False
    try:
        result = experiment.main(ctx)
        complete = True
    except Pending as p:
        return False, str(p)
    finally:
        # Persist the requested set (even on Pending/TaskFailed) so read-only views
        # can split current records from superseded ones. ``main`` replays from the
        # top each wake, so the set is complete up to the suspension point; keys past
        # it aren't known yet and their old records read as superseded until a later
        # wake requests them again — honest, since an upstream edit may have changed
        # what they'll ask for. ``kept_stale`` rides along so read-only views can
        # badge results served under old code (they can't fingerprint code
        # themselves — reads never import or tick). ``complete`` records whether
        # this manifest covers the *whole* DAG (``main`` ran to the end) — the
        # gate ``mini gc`` needs before trusting "not requested" to mean
        # "never requested again".
        store.set_meta(
            requested=list(dict.fromkeys(ctx.requested)),
            kept_stale=list(dict.fromkeys(ctx.stale_kept)),
            complete=complete,
        )
    return True, result


def retry(store: MemoStore, key: str | None = None) -> list[str]:
    """Reset settled-but-not-DONE tasks so the next ``tick`` reruns them.

    ``FAILED``/``CANCELLED`` are terminal *under the code that produced them* —
    ``tick`` won't auto-relaunch them — so re-running takes intent: this, or
    editing the fn (stale evidence relaunches on the next tick), or bumping
    ``version=``. This clears their records (state → un-run, prior attempt kept
    in history); the next ``tick`` then relaunches. Pass *key* to retry one task;
    otherwise all failed/cancelled tasks. Returns the keys reset (a `DONE` task
    is never reset — edit the fn or bump ``version=`` to force that).

    Superseded records — keys the last tick no longer requested (their fn was
    renamed, their config removed) — are skipped: no tick will ever relaunch them,
    so resetting one just plants a phantom forever-pending record. An explicit
    *key* overrides (deliberate intent beats the manifest).
    """
    requested = store.requested_keys()
    targets: list[str] = []
    for rec in store.records():
        k = rec["key"]
        if key is not None and k != key:
            continue
        if key is None and requested is not None and k not in requested:
            continue  # superseded — the DAG no longer requests this key
        state = RunState(rec["state"]) if rec.get("state") else None
        if state in (RunState.FAILED, RunState.CANCELLED):
            store.reset(k)
            targets.append(k)
    return targets

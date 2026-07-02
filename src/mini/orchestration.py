"""
Memoized orchestration for multi-step experiments.

An experiment is a plain function ``main(ctx)`` that expresses the DAG in
ordinary Python. Each ``ctx.run`` / ``ctx.map`` is content-addressed: cached ->
return the stored result; otherwise launch a detached task and *suspend* the
wake by raising ``Pending``. A driver re-runs ``main`` each wake; completed steps
are memo hits, so only the un-run / failed pieces execute — crash-recovery by
re-run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, overload

from mini.memo import MemoStore, fingerprint
from mini.runs import SETTLED, RunState

if TYPE_CHECKING:
    from mini.apparatus import Apparatus
    from mini.experiment import Experiment

__all__ = ['MemoError', 'Pending', 'TaskFailed', 'BudgetExpired', 'MISSING', 'Ctx', 'tick', 'retry']


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
        super().__init__(f'wall-clock budget elapsed — cancelled {len(cancelled)} in-flight task(s)')


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

    def __init__(self, key: str, state: RunState, error: str = '', exc_type: str = ''):
        self.key = key
        self.state = state
        self.error = error
        self.exc_type = exc_type
        head = f'{key} settled {state}'
        super().__init__(f'{head}\n{error}' if error and error != '(no logs)' else head)


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
        return '<missing>'

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

    def __init__(self, store: MemoStore, apparatus: Apparatus, roles: dict[str, Apparatus] | None = None):
        self.store = store
        self.apparatus = apparatus
        self.roles = roles or {}
        self.launched: list[str] = []
        self.pending: list[str] = []
        # Every key this wake resolved (hit, in-flight, or launched) — the DAG's
        # *requested set*. A record whose key a wake no longer requests (its fn was
        # edited, its config removed) is superseded: still on disk, but not part of
        # the run's state. ``tick`` persists this so read-only views can tell the two
        # apart without re-running ``main``.
        self.requested: list[str] = []

    def _route(self, on: Apparatus | None, role: str | None) -> Apparatus:
        """Resolve which apparatus a step runs on: ``role`` label, ``on=``, or default."""
        if on is not None and role is not None:
            raise ValueError('pass either role= or on=, not both')
        if role is not None:
            if role not in self.roles:
                known = ', '.join(sorted(self.roles)) or '(none defined)'
                raise ValueError(f'unknown role {role!r}; defined roles: {known}')
            return self.roles[role]
        return on or self.apparatus

    def _classify(
        self, fn: Callable, args: tuple, version: str | None, app: Apparatus
    ) -> tuple[str, RunState | None, tuple[str, Callable, tuple, list] | None]:
        """Resolve a call's key/state and, if it needs launching, mark it RUNNING
        and return its batch entry — *without* spawning. The caller batches the
        spawn so a ``map`` fans out in one ``spawn_tasks`` call.
        """
        key = fingerprint(fn, args, version)
        self.requested.append(key)
        state = self.store.state(key)
        to_launch: tuple[str, Callable, tuple, list] | None = None
        if state is None:  # never run; FAILED/CANCELLED are terminal (retry takes intent)
            self.store.mark_running(fn, key)
            to_launch = (key, fn, args, getattr(app, '_before_hooks', []))
            self.launched.append(key)
            state = RunState.RUNNING
        return key, state, to_launch

    def _task_failed(self, key: str, state: RunState) -> TaskFailed:
        """Build a ``TaskFailed`` for a settled-but-not-DONE key, with its stored traceback + type."""
        return TaskFailed(key, state, self.store.error(key), self.store.record(key).get('exc_type', ''))

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
        raise Pending(f'waiting on {key}')

    @overload
    def map[R](
        self,
        fn: Callable[..., R],
        items: Sequence[Any],
        version: str | None = None,
        on: Apparatus | None = None,
        role: str | None = None,
        allow_partial: Literal[False] = False,
    ) -> list[R]: ...
    @overload
    def map[R](
        self,
        fn: Callable[..., R],
        items: Sequence[Any],
        version: str | None = None,
        on: Apparatus | None = None,
        role: str | None = None,
        *,
        allow_partial: Literal[True],
    ) -> list[R | _Missing]: ...
    def map[R](
        self,
        fn: Callable[..., R],
        items: Sequence[Any],
        version: str | None = None,
        on: Apparatus | None = None,
        role: str | None = None,
        allow_partial: bool = False,
    ) -> list[R] | list[R | _Missing]:
        """Fan out *fn* over *items*, suspending until the results are ready.

        Launches every missing item in one batch, then raises ``Pending`` while
        any task is still in flight. By default the map is all-or-nothing: once the
        fan-out has *settled*, any cell that settled ``FAILED``/``CANCELLED`` raises
        — all of them together, as an ``ExceptionGroup`` of ``TaskFailed`` (so you
        see every failure, not just the first). ``tick`` won't relaunch a terminal
        cell, so this is the DAG giving up rather than spinning; ``retry`` heals it.

        With ``allow_partial=True`` the map still waits for in-flight tasks, but
        once everything has settled it returns instead of raising: the result list
        stays index-aligned with *items*, with ``MISSING`` in the position of each
        failed/cancelled cell. This lets the pipeline's later steps run on the
        subset that succeeded.
        """
        app = self._route(on, role)
        results: list[Any] = []
        batch: list[tuple[str, Callable, tuple, list]] = []
        failed: list[tuple[str, RunState]] = []
        # `self.pending` holds only truly in-flight keys; settled-but-failed cells go
        # to `failed`. Ctx is rebuilt each tick and the first incomplete step raises,
        # so this stays a clean per-tick view across steps.
        for raw in items:
            args = raw if isinstance(raw, tuple) else (raw,)
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
            raise Pending(f'{len(self.pending)} task(s) in flight')
        if failed and not allow_partial:  # everything settled — surface the failures together
            raise ExceptionGroup(
                f'{len(failed)} of {len(items)} task(s) failed',
                [self._task_failed(key, state) for key, state in failed],
            )
        return results


def tick(experiment: Experiment, apparatus: Apparatus) -> tuple[bool, Any]:
    """Run one wake of an experiment's orchestration on *apparatus*.

    Returns ``(done, payload)``: ``(True, result)`` if the DAG completed, or
    ``(False, reason)`` if it suspended waiting on in-flight tasks. Propagates
    ``TaskFailed`` (or an ``ExceptionGroup`` of them, from a strict ``map``) when a
    step the DAG depends on has settled terminally — the run can't progress without
    a ``retry``. Steps can override the apparatus per call via ``ctx.run(...,
    on=)`` / ``ctx.map(..., on=)``.
    """
    store = apparatus.memo_store()
    ctx = Ctx(store, apparatus, experiment.resolve_roles(apparatus))
    try:
        result = experiment.orchestration()(ctx)
    except Pending as p:
        return False, str(p)
    finally:
        # Persist the requested set (even on Pending/TaskFailed) so read-only views
        # can split current records from superseded ones. ``main`` replays from the
        # top each wake, so the set is complete up to the suspension point; keys past
        # it aren't known yet and their old records read as superseded until a later
        # wake requests them again — honest, since an upstream edit may re-key them.
        store.set_meta(requested=list(dict.fromkeys(ctx.requested)))
    return True, result


def retry(store: MemoStore, key: str | None = None) -> list[str]:
    """Reset settled-but-not-DONE tasks so the next ``tick`` reruns them.

    ``FAILED``/``CANCELLED`` are terminal — ``tick`` won't auto-relaunch them — so
    re-running takes intent. This clears their records (state → un-run); the next
    ``tick`` then relaunches. Pass *key* to retry one task; otherwise all
    failed/cancelled tasks. Returns the keys reset (a `DONE` task is never reset —
    edit the fn or bump ``version=`` to force that). DONE results stay memo hits.

    Superseded records — keys the last tick no longer requested (their fn was
    edited, their config removed) — are skipped: no tick will ever relaunch them,
    so resetting one just plants a phantom forever-pending record. An explicit
    *key* overrides (deliberate intent beats the manifest).
    """
    requested = store.requested_keys()
    targets: list[str] = []
    for rec in store.records():
        k = rec['key']
        if key is not None and k != key:
            continue
        if key is None and requested is not None and k not in requested:
            continue  # superseded — the DAG no longer requests this key
        state = RunState(rec['state']) if rec.get('state') else None
        if state in (RunState.FAILED, RunState.CANCELLED):
            store.reset(k)
            targets.append(k)
    return targets

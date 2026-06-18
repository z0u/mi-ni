"""
Memoized orchestration for multi-step experiments.

An experiment is a plain function ``main(ctx)`` that expresses the DAG in
ordinary Python. Each ``ctx.run`` / ``ctx.map`` is content-addressed: cached ->
return the stored result; otherwise launch a detached task and *suspend* the
wake by raising ``Pending``. A driver re-runs ``main`` each wake; completed steps
are memo hits, so only the un-run / failed pieces execute — crash-recovery by
re-run. See notes/agentic-experiments.md (section 5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from mini.memo import MemoStore, fingerprint
from mini.runs import RunState

if TYPE_CHECKING:
    from mini.apparatus import Apparatus
    from mini.experiment import Experiment

__all__ = ['Pending', 'Ctx', 'tick', 'retry']


class Pending(Exception):
    """Raised to suspend the current wake until in-flight tasks finish."""


class Ctx:
    """The memoized, non-blocking ``run``/``map`` an orchestration calls.

    ``run`` raises ``Pending`` the moment a result isn't ready, so code after it
    only runs once the result exists. For parallel fan-out use ``map``, which
    launches *all* missing tasks before suspending.

    Each call runs on the tick's *apparatus* by default; pass ``on=`` to route a
    step elsewhere (e.g. CPU prep, GPU training). The apparatus also supplies the
    per-step ``before_each`` hooks.
    """

    def __init__(self, store: MemoStore, apparatus: Apparatus):
        self.store = store
        self.apparatus = apparatus
        self.launched: list[str] = []
        self.pending: list[str] = []

    def _classify(
        self, fn: Callable, args: tuple, version: str | None, app: Apparatus
    ) -> tuple[str, RunState | None, tuple[str, Callable, tuple, list] | None]:
        """Resolve a call's key/state and, if it needs launching, mark it RUNNING
        and return its batch entry — *without* spawning. The caller batches the
        spawn so a ``map`` fans out in one ``spawn_tasks`` call.
        """
        key = fingerprint(fn, args, version)
        state = self.store.state(key)
        to_launch: tuple[str, Callable, tuple, list] | None = None
        if state is None:  # never run; FAILED/CANCELLED are terminal (retry takes intent)
            self.store.mark_running(fn, key)
            to_launch = (key, fn, args, getattr(app, '_before_hooks', []))
            self.launched.append(key)
            state = RunState.RUNNING
        return key, state, to_launch

    def run(self, fn: Callable, *args: Any, version: str | None = None, on: Apparatus | None = None) -> Any:
        app = on or self.apparatus
        key, state, to_launch = self._classify(fn, args, version, app)
        if to_launch is not None:
            app.spawn_tasks(self.store, [to_launch])
        if state == RunState.DONE:
            return self.store.result(key)
        self.pending.append(key)
        raise Pending(f'waiting on {key}')

    def map(
        self, fn: Callable, items: Sequence[Any], version: str | None = None, on: Apparatus | None = None
    ) -> list[Any]:
        app = on or self.apparatus
        results: list[Any] = []
        batch: list[tuple[str, Callable, tuple, list]] = []
        for raw in items:
            args = raw if isinstance(raw, tuple) else (raw,)
            key, state, to_launch = self._classify(fn, args, version, app)
            if to_launch is not None:
                batch.append(to_launch)
            if state == RunState.DONE:
                results.append(self.store.result(key))
            else:
                self.pending.append(key)
        if batch:  # one spawn for the whole fan-out
            app.spawn_tasks(self.store, batch)
        if self.pending:
            raise Pending(f'{len(self.pending)} task(s) in flight')
        return results


def tick(experiment: Experiment, apparatus: Apparatus) -> tuple[bool, Any]:
    """Run one wake of an experiment's orchestration on *apparatus*.

    Returns ``(done, payload)``: ``(True, result)`` if the DAG completed, or
    ``(False, reason)`` if it suspended waiting on in-flight tasks. Steps can
    override the apparatus per call via ``ctx.run(..., on=)`` / ``ctx.map(...,
    on=)``.
    """
    store = apparatus.memo_store()
    ctx = Ctx(store, apparatus)
    try:
        result = experiment.orchestration()(ctx)
    except Pending as p:
        return False, str(p)
    return True, result


def retry(store: MemoStore, key: str | None = None) -> list[str]:
    """Reset settled-but-not-DONE tasks so the next ``tick`` reruns them.

    ``FAILED``/``CANCELLED`` are terminal — ``tick`` won't auto-relaunch them — so
    re-running takes intent. This clears their records (state → un-run); the next
    ``tick`` then relaunches. Pass *key* to retry one task; otherwise all
    failed/cancelled tasks. Returns the keys reset (a `DONE` task is never reset —
    edit the fn or bump ``version=`` to force that). DONE results stay memo hits.
    """
    targets: list[str] = []
    for rec in store.records():
        k = rec['key']
        if key is not None and k != key:
            continue
        state = RunState(rec['state']) if rec.get('state') else None
        if state in (RunState.FAILED, RunState.CANCELLED):
            store.reset(k)
            targets.append(k)
    return targets

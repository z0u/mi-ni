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
    from mini.experiment import Experiment

__all__ = ['Pending', 'Ctx', 'tick']


class Pending(Exception):
    """Raised to suspend the current wake until in-flight tasks finish."""


class Ctx:
    """The memoized, non-blocking ``run``/``map`` an orchestration calls.

    ``run`` raises ``Pending`` the moment a result isn't ready, so code after it
    only runs once the result exists. For parallel fan-out use ``map``, which
    launches *all* missing tasks before suspending.
    """

    def __init__(self, store: MemoStore, before_hooks: list[Callable] | None = None):
        self.store = store
        self._hooks = before_hooks or []
        self.launched: list[str] = []
        self.pending: list[str] = []

    def _ensure(self, fn: Callable, args: tuple, version: str | None) -> tuple[str, RunState | None]:
        key = fingerprint(fn, args, version)
        state = self.store.state(key)
        if state in (None, RunState.FAILED):
            self.store.launch(fn, args, key, hooks=self._hooks)
            self.launched.append(key)
            state = RunState.RUNNING
        return key, state

    def run(self, fn: Callable, *args: Any, version: str | None = None) -> Any:
        key, state = self._ensure(fn, args, version)
        if state == RunState.DONE:
            return self.store.result(key)
        self.pending.append(key)
        raise Pending(f'waiting on {key}')

    def map(self, fn: Callable, items: Sequence[Any], version: str | None = None) -> list[Any]:
        results: list[Any] = []
        for raw in items:
            args = raw if isinstance(raw, tuple) else (raw,)
            key, state = self._ensure(fn, args, version)
            if state == RunState.DONE:
                results.append(self.store.result(key))
            else:
                self.pending.append(key)
        if self.pending:
            raise Pending(f'{len(self.pending)} task(s) in flight')
        return results


def tick(experiment: Experiment) -> tuple[bool, Any]:
    """Run one wake of an experiment's orchestration.

    Returns ``(done, payload)``: ``(True, result)`` if the DAG completed, or
    ``(False, reason)`` if it suspended waiting on in-flight tasks.
    """
    store = MemoStore(experiment.data_dir())
    ctx = Ctx(store, before_hooks=experiment.before_hooks())
    try:
        result = experiment.orchestration()(ctx)
    except Pending as p:
        return False, str(p)
    return True, result

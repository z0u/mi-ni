"""
Executor-like protocol that abstracts compute and storage.
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Generic, Iterable, Iterator, ParamSpec, TypeVar

from mini.volume import Volume

if TYPE_CHECKING:
    from mini.gc import GcIO
    from mini.memo import MemoStore
    from mini.store import Store

P = ParamSpec("P")
R = TypeVar("R")
V = TypeVar("V", bound=Volume)

# Persistent background event loop shared across sync-from-async calls.
# A single loop avoids the problem where frameworks like Modal track state
# per-loop and don't reset when an ``asyncio.run()`` loop is destroyed.
_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None
_bg_lock = threading.Lock()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    """Return (and lazily start) a long-lived background event loop."""
    global _bg_loop, _bg_thread
    with _bg_lock:
        if _bg_loop is None or _bg_loop.is_closed():
            _bg_loop = asyncio.new_event_loop()
            _bg_thread = threading.Thread(
                target=_bg_loop.run_forever,
                daemon=True,
            )
            _bg_thread.start()
        return _bg_loop


# ---------------------------------------------------------------------------
# Apparatus protocol
# ---------------------------------------------------------------------------


class Apparatus(ABC, Generic[V]):
    """Protocol for running a function over a sweep of inputs."""

    _volume: V | None

    @property
    def volume(self) -> V:
        """Return the volume; raises ``RuntimeError`` if none is configured."""
        if self._volume is None:
            raise RuntimeError("No volume configured for this apparatus. Set .volume before accessing it.")
        return self._volume

    @volume.setter
    def volume(self, value: V | None) -> None:
        self._volume = value

    def run(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Run a single function and return its result."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — call arun directly.
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.arun(fn, *args, **kwargs))  # pyrefly: ignore [bad-argument-type]
            finally:
                loop.close()

        # Running loop detected — offload to background loop.
        future = asyncio.run_coroutine_threadsafe(
            self.arun(fn, *args, **kwargs),  # pyrefly: ignore [bad-argument-type]
            _get_background_loop(),
        )
        return future.result()

    async def arun(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Run a single function and return its result, asynchronously."""

        @wraps(fn)
        def wrapper(_) -> R:
            return fn(*args, **kwargs)

        results = [r async for r in self.amap(wrapper, [None])]
        return results[0]

    @abstractmethod
    def amap(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[R, None]:
        """
        Map *fn* over one or more iterables.

        Like ``concurrent.futures.Executor.map`` and Modal's ``Function.map``:
        the iterables are zipped together and each tuple is unpacked as
        positional arguments.  *kwargs* (if given) are forwarded to every
        call.

        ::

            app.map(fn, [1, 2, 3])                    # fn(1), fn(2), fn(3)
            app.map(fn, [1, 2], ['a', 'b'])            # fn(1, 'a'), fn(2, 'b')
            app.map(fn, [1, 2], kwargs={'k': 'v'})     # fn(1, k='v'), fn(2, k='v')
        """
        ...

    def map(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> Iterator[R]:
        """
        Map *fn* over one or more iterables.

        Like ``concurrent.futures.Executor.map`` and Modal's ``Function.map``:
        the iterables are zipped together and each tuple is unpacked as
        positional arguments.  *kwargs* (if given) are forwarded to every
        call.

        ::

            app.map(fn, [1, 2, 3])                    # fn(1), fn(2), fn(3)
            app.map(fn, [1, 2], ['a', 'b'])            # fn(1, 'a'), fn(2, 'b')
            app.map(fn, [1, 2], kwargs={'k': 'v'})     # fn(1, k='v'), fn(2, k='v')
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            yield from _map_with_new_loop(self, fn, *iterables, kwargs=kwargs)
            return

        yield from _map_in_thread(self, fn, *iterables, kwargs=kwargs)

    def w(self, **kwargs: Any) -> Apparatus:
        """Return a variant with backend-native options applied (e.g. ``gpu='L4'``).

        Role resolution calls this to specialise one base apparatus per role.
        The default ignores all options and returns ``self``: a local backend
        has no extra knobs, so a role table written for Modal still loads locally.
        ``ModalApparatus`` overrides this to merge ``@function`` kwargs.
        """
        return self

    @abstractmethod
    def before_each(self, hook: Callable[[], Any]) -> Apparatus:
        """Return a new apparatus that runs *hook* before each job.

        Useful for per-job setup like configuring logging or seeding RNGs.
        *hook* takes no arguments; its return value is ignored.
        """
        ...

    # -- Detached, memoized orchestration -------------------------------------
    # Unlike map/amap (blocking launch + monitor + collect for notebooks), the
    # memoized path splits the lifecycle across short-lived processes: tick
    # stages each call and spawn_tasks launches it detached.

    def store(self) -> Store:
        """Return the content-addressed artifact :class:`~mini.store.Store` for this backend.

        Distinct from :meth:`memo_store` (the per-experiment control plane): the
        artifact store is **project-scoped**, so blobs and named refs are shared
        across experiments — content-addressed, so identical bytes coincide. The
        default sits a ``store/`` beside the experiment's volume (the project
        root); ``ModalApparatus`` overrides it to read blobs back off the Volume.
        Both the apparatus (here, for reports) and the worker enter this around a
        step, so ``mini.store.put`` / ``get`` resolve against the same store.
        """
        from mini.store import store_for, store_root_for

        return store_for(store_root_for(self.volume.path))

    @abstractmethod
    def memo_store(self) -> MemoStore:
        """Return the ``MemoStore`` for memoized orchestration on this backend.

        Binds the record store (small/hot state) to the volume (results).
        Each backend constructs its own: local uses JSON files; Modal uses a
        ``modal.Dict`` for records and reads results back from the Volume.
        Constructing it here (rather than at call sites) lets ``tick`` stay
        backend-agnostic.
        """

    def gc_io(self, store: MemoStore) -> GcIO:
        """The I/O-plane adapter ``mini gc`` sweeps this backend through.

        Local result dirs are plain files under the store's ``data_dir``;
        ``ModalApparatus`` overrides this to sweep the Volume by path instead.
        """
        from mini.gc import LocalGcIO

        return LocalGcIO(store)
        ...

    @abstractmethod
    def spawn_tasks(self, store: MemoStore, batch: list[tuple[str, str, Callable, tuple, list]]) -> None:
        """Spawn detached workers for a batch of memoized tasks.

        ``Ctx`` claims each record RUNNING under a fresh generation, then passes
        the batch — each entry ``(key, gen, fn, args, hooks)`` — here to launch
        workers that persist results under each key, surviving the tick that
        launched them. *gen* travels with the call: the worker fences all its
        writes on it. Batching lets ``ctx.map`` fan out efficiently (one
        ``spawn_map`` on Modal rather than one detached call per task).
        """
        ...

    def cancel(self, store: MemoStore) -> list[str]:
        """Stop in-flight tasks, mark them CANCELLED, and return their keys.

        Delegates per-task stops to ``_stop_task`` (local SIGTERMs the worker
        process group; Modal cancels the ``FunctionCall``). Settled tasks are
        left alone. Releasing ``gen`` fences the worker even if it survives the
        stop (an ignored SIGTERM): its writes no longer own the record, so it
        can't flip CANCELLED back to DONE and pass a half-cancelled attempt off
        as a current result.
        """
        from mini.runs import RunState

        cancelled: list[str] = []
        for rec in store.records():
            state = RunState(rec["state"]) if rec.get("state") else RunState.PENDING
            if state in (RunState.RUNNING, RunState.PENDING):
                self._stop_task(rec)
                store.update(rec["key"], state=RunState.CANCELLED, gen=None)
                cancelled.append(rec["key"])
        return cancelled

    def enforce_budget(self, store: MemoStore) -> list[str]:
        """Tear the run down if its wall-clock (cost) budget has elapsed; return cancelled keys.

        A detached run has no supervising process, so a forgotten or wedged sweep
        can burn money on Modal — or hold local resources — indefinitely. A budget
        stamps a ``deadline_at`` into the control plane at launch; any process that
        already polls the store (``status`` / ``watch`` / the ``--watch`` driver)
        calls this to enforce it *opportunistically*, settling in-flight tasks
        CANCELLED via the existing ``cancel`` path (local SIGTERM / Modal
        ``FunctionCall.cancel``). A no-op before the deadline, or when no budget is
        set, so it's safe to call on every read/poll path.
        """
        return self.cancel(store) if store.budget_expired() else []

    def _stop_task(self, rec: dict[str, Any]) -> None:
        """Backend-specific: stop one in-flight task. Default: nothing to stop."""

    def reap_dead(self, store: MemoStore, records: list[dict[str, Any]] | None = None) -> list[str]:
        """Settle RUNNING tasks whose worker has vanished (→ FAILED); return their keys.

        A killed or crashed worker can exit without writing a settled state,
        leaving a stale RUNNING record that wedges ``--watch`` forever. We
        cross-check each RUNNING task via ``_is_task_alive`` and mark orphans
        FAILED — recovery then requires a deliberate ``retry``. Reaping never
        relaunches, so it's safe on the read/poll path.

        Pass *records* to reuse a snapshot already in hand (avoiding a second
        full read); reaped records are mutated in place so the caller's copy
        stays current.
        """
        from mini.runs import RunState

        reaped: list[str] = []
        for rec in store.records() if records is None else records:
            if rec.get("state") != RunState.RUNNING or self._is_task_alive(rec):
                continue
            # Re-read before settling: a worker writes its final state *then* exits,
            # so if it's gone yet the record still says RUNNING it died mid-run. The
            # re-read closes the gap between our records() snapshot and the probe.
            if store.state(rec["key"]) != RunState.RUNNING:
                continue
            error = "worker vanished (killed/crashed, no result written)"
            # gen=None releases the attempt: if the liveness probe was wrong and the
            # worker still breathes somewhere, its fenced writes can't undo the reap.
            store.update(rec["key"], state=RunState.FAILED, error=error, gen=None)
            rec["state"], rec["error"] = RunState.FAILED, error  # keep the caller's snapshot current
            reaped.append(rec["key"])
        return reaped

    def _is_task_alive(self, rec: dict[str, Any]) -> bool:
        """Is this RUNNING task's worker still alive?

        Defaults to ``True`` (unknown → alive): a backend with no liveness probe
        never reaps a task it can't confirm is dead. False negatives (marking a
        live task dead) are far more harmful than false positives (letting a
        stale record linger).
        """
        return True


def _map_in_thread(
    app: Apparatus,
    fn: Callable[..., R],
    *iterables: Iterable[Any],
    kwargs: dict[str, Any] | None,
) -> Iterator[R]:
    import queue as queue_module

    results_queue: queue_module.Queue = queue_module.Queue()

    async def collect():
        try:
            async for result in app.amap(fn, *iterables, kwargs=kwargs):
                results_queue.put(("result", result))
            results_queue.put(("done", None))
        except Exception as e:
            results_queue.put(("error", e))

    future = asyncio.run_coroutine_threadsafe(collect(), _get_background_loop())

    while True:
        msg_type, value = results_queue.get()
        if msg_type == "result":
            yield value
        elif msg_type == "done":
            break
        elif msg_type == "error":
            raise value

    # Ensure the coroutine finished cleanly.
    future.result()


def _map_with_new_loop(
    app: Apparatus,
    fn: Callable[..., R],
    *iterables: Iterable[Any],
    kwargs: dict[str, Any] | None,
) -> Iterator[R]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        gen = app.amap(fn, *iterables, kwargs=kwargs)
        while True:
            try:
                yield loop.run_until_complete(gen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()

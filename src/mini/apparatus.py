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
    from mini.memo import MemoStore
    from mini.runs import Run

P = ParamSpec('P')
R = TypeVar('R')
V = TypeVar('V', bound=Volume)

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
        """
        Return the volume.

        Raises ``RuntimeError`` if no volume is configured.
        """
        if self._volume is None:
            # Raise instead of returning None: accessing the volume when none is
            # configured is exceptional, and None complicates the types.
            raise RuntimeError('No volume configured for this apparatus. Set .volume before accessing it.')
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

    @abstractmethod
    def before_each(self, hook: Callable[[], Any]) -> Apparatus:
        """
        Return a new apparatus that runs *hook* before each job.

        This is useful for things like configuring logging or setting random
        seeds on a per-job basis.

        Arguments:
            hook: A function to run before each job. It should take no
            arguments. Its return value is ignored.
        """
        ...

    # -- Detached lifecycle ---------------------------------------------------
    # Unlike ``map``/``amap`` (launch + monitor + collect in one blocking call),
    # these split the lifecycle so it can span short-lived processes: ``submit``
    # launches detached and returns a durable handle; ``reopen`` reconstructs it
    # elsewhere. See notes/agentic-experiments.md.

    def submit(self, fn: Callable[..., R], *iterables: Iterable[Any], kwargs: dict[str, Any] | None = None) -> Run:
        """Launch *fn* over the iterables detached; return a durable `Run`."""
        raise NotImplementedError(
            f'{type(self).__name__} does not support detached submit yet. See notes/agentic-experiments.md.'
        )

    def reopen(self, run_id: str) -> Run:
        """Reconstruct a `Run` handle from a previous submit, in a fresh process."""
        raise NotImplementedError(f'{type(self).__name__} does not support reopen yet.')

    def memo_store(self) -> MemoStore:
        """Return the `MemoStore` for the memoized orchestration on this backend.

        The store binds the two planes the backend uses: a ``RecordStore`` for
        small/hot state, and the Volume for results. The default is fully local
        (JSON records + local files under the Volume path); Modal overrides this
        to put records in a ``modal.Dict`` and read results back from the Volume.
        Constructing it here (rather than ``MemoStore(volume.path)`` at call
        sites) is what lets ``tick`` stay backend-agnostic.
        """
        from mini.memo import MemoStore

        return MemoStore(self.volume.path)

    def spawn_tasks(self, store: MemoStore, batch: list[tuple[str, Callable, tuple, list]]) -> None:
        """Spawn detached workers for a *batch* of memoized tasks, on this apparatus.

        The seam that lets the memoized orchestration (``mini.orchestration.Ctx``)
        decide *where* steps run. ``Ctx`` marks each record RUNNING
        (``MemoStore.mark_running``), then hands the batch — each entry
        ``(key, fn, args, hooks)`` — to this method, which launches workers that
        run the calls and persist their results/state under each *key*, surviving
        the tick that launched them.

        Batching is what lets a ``ctx.map`` fan out efficiently: the local backend
        spawns one subprocess per task, while Modal issues a single ``spawn_map``
        rather than one detached ``app.run`` per task. ``ctx.run`` passes a
        one-element batch. How the call reaches the worker is backend-specific
        (staged to disk vs handed to ``spawn``); either way per-step ``on=`` routes
        *compute*, not just hooks.
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not support detached spawn_tasks yet. See notes/agentic-experiments.md.'
        )


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
                results_queue.put(('result', result))
            results_queue.put(('done', None))
        except Exception as e:
            results_queue.put(('error', e))

    future = asyncio.run_coroutine_threadsafe(collect(), _get_background_loop())

    while True:
        msg_type, value = results_queue.get()
        if msg_type == 'result':
            yield value
        elif msg_type == 'done':
            break
        elif msg_type == 'error':
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

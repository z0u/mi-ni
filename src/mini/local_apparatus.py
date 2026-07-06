"""
Apparatus for running sweeps locally with thread-based concurrency.

Example::

    from mini.local_apparatus import LocalApparatus

    app = LocalApparatus("my-experiment", max_workers=4)
    results = list(app.map(train, configs))
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import signal
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Iterable, TypeVar, override

from mini._queues import QueueLike
from mini.apparatus import Apparatus
from mini.local_queue import LocalQueue
from mini.local_volume import LocalVolume
from mini.memo import MemoStore
from mini.progress import ProgressMessage, progress_context
from mini.progress_display import RichProgressDisplay
from mini.runs import data_root, spawn_taskworker
from mini.store import Store, project_store, store_context, store_for, store_root_for
from mini.volume import data_dir_context

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["LocalApparatus"]


class LocalApparatus(Apparatus[LocalVolume]):
    """
    Run functions locally using a thread pool.

    Jobs can report progress via ``emit_progress()`` which is automatically
    displayed using Rich progress bars when running in a terminal.
    """

    def __init__(self, name: str, max_workers: int = 1, data_dir: Path | str | None = None):
        self.name = name
        self.max_workers = max_workers
        self._before_hooks: list[Callable[[], Any]] = []
        self._volume: LocalVolume | None = LocalVolume(Path(data_dir) if data_dir else data_root() / name)

    def __str__(self) -> str:
        return f'Local apparatus "{self.name}"'

    def clone(self) -> LocalApparatus:
        new_app = LocalApparatus(self.name, self.max_workers)
        new_app._before_hooks = self._before_hooks[:]
        new_app._volume = self._volume
        return new_app

    @override
    def before_each(self, hook: Callable[[], Any]) -> LocalApparatus:
        new_app = self.clone()
        new_app._before_hooks = self._before_hooks + [hook]
        return new_app

    @override
    def memo_store(self) -> MemoStore:
        from mini.memo import MemoStore

        return MemoStore(self.volume.path)

    @override
    def spawn_tasks(self, store: MemoStore, batch: list[tuple[str, str, Callable, tuple, list]]) -> None:
        for key, gen, fn, args, hooks in batch:
            store.write_call(key, fn, args, hooks, gen)  # stage to disk for the subprocess worker
            store.update_if(key, gen, pid=spawn_taskworker(store.data_dir, key))  # pid == pgid, for cancel

    @override
    def _stop_task(self, rec: dict[str, Any]) -> None:
        """SIGTERM the worker's process group (it's a session leader: pgid == pid)."""
        if pid := rec.get("pid"):
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGTERM)

    @override
    def _is_task_alive(self, rec: dict[str, Any]) -> bool:
        """Is the recorded worker pid still a live process? (for ``reap_dead``)."""
        pid = rec.get("pid")
        return _pid_alive(pid) if pid else True  # no pid yet — can't probe; assume alive

    @override
    async def amap(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[R, None]:
        # TODO: support lazy iterables
        iterables_lists: list[list] = [list(it) for it in iterables]
        sizes = [len(it) for it in iterables_lists]
        n = min(sizes) if sizes else None

        # Name the backend (symmetric with Modal's 'Running … on Modal'), so a
        # local run — e.g. a fallback when a notebook meant to use Modal — is
        # visible in the logs rather than only inferable from the *absence* of
        # Modal's image-build output. ('locally', not 'on CPU': a local box may
        # well have a GPU that JAX/torch will use.)
        log.info("Running %d jobs locally (%d workers)", n, self.max_workers)
        run_id = secrets.token_hex(4)

        if self._volume is not None:
            self._volume.path.mkdir(parents=True, exist_ok=True)

        progress_display = RichProgressDisplay(n or 0, queue=LocalQueue())
        # Target ~10 emissions/sec overall: interval = max_workers / target_rate_hz
        emission_interval = self.max_workers / 10.0
        # Project-scoped artifact store, so a mapped fn's put/get resolves the ambient
        # store on the interactive path too (not only the detached memo worker). Built
        # caller-side and closed over: local execution is in-process threads.
        store = store_for(store_root_for(self._volume.path)) if self._volume is not None else project_store()
        local_fn = _wrap_for_local(
            fn,
            self._before_hooks,
            run_id,
            progress_display.queue,
            kwargs=kwargs or {},
            emission_interval=emission_interval,
            data_dir=self._volume.path if self._volume is not None else None,
            store=store,
        )

        loop = asyncio.get_running_loop()

        with progress_display, ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # Submit all tasks
            tasks = [
                loop.run_in_executor(pool, local_fn, i, *args)
                for i, args in enumerate(zip(*iterables_lists, strict=False))
            ]

            # Yield results in input order to match map semantics
            for task in tasks:
                yield await task


def _pid_alive(pid: int) -> bool:
    """Whether *pid* is a running process — counting a zombie as *not* alive.

    ``os.kill(pid, 0)`` succeeds on a zombie (an exited child not yet reaped),
    which would keep a hard-killed worker looking alive when it's a direct child
    of the watcher. On Linux we read ``/proc/<pid>/stat`` and treat state ``Z`` as
    dead; elsewhere we fall back to a signal-0 probe (no zombie distinction).
    """
    proc = Path("/proc") / str(pid)
    if Path("/proc").is_dir():
        if not proc.exists():
            return False
        try:
            # stat is "pid (comm) state ..."; comm may hold spaces/parens, so the
            # state field is the first token after the final ')'.
            return (proc / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except OSError:
            return False  # vanished between the exists() check and the read
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours (shouldn't happen for our own worker)


def _wrap_for_local(
    fn: Callable[..., R],
    hooks: list[Callable[[], None]],
    run_id: str,
    queue: QueueLike[ProgressMessage],
    kwargs: dict[str, Any],
    emission_interval: float,
    data_dir: Path | None,
    store: Store | None,
) -> Callable[..., R]:
    def run_one(index: int, *args) -> R:
        dir_ctx = data_dir_context(path=data_dir) if data_dir is not None else nullcontext()
        store_ctx = store_context(store) if store is not None else nullcontext()
        with (
            progress_context(run_id, str(index), queue=queue, emission_interval=emission_interval),
            dir_ctx,
            store_ctx,
        ):
            for hook in reversed(hooks):
                hook()
            result = fn(*args, **kwargs)
            return result

    return run_one

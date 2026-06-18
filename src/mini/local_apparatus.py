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
import secrets
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Iterable, TypeVar, override

from mini._queues import QueueLike
from mini.apparatus import Apparatus
from mini.local_queue import LocalQueue
from mini.local_volume import LocalVolume
from mini.memo import MemoStore
from mini.progress import ProgressMessage, progress_context
from mini.progress_display import RichProgressDisplay
from mini.runs import spawn_taskworker
from mini.volume import data_dir_context

log = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

__all__ = ['LocalApparatus']


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
        self._volume: LocalVolume | None = LocalVolume(Path(data_dir) if data_dir else Path(f'.mini/{name}'))

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
    def spawn_tasks(self, store: MemoStore, batch: list[tuple[str, Callable, tuple, list]]) -> None:
        for key, fn, args, hooks in batch:
            store.write_call(key, fn, args, hooks)  # stage to disk for the subprocess worker
            spawn_taskworker(store.data_dir, key)

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

        log.info('Running %d jobs with %d workers', n, self.max_workers)
        run_id = secrets.token_hex(4)

        if self._volume is not None:
            self._volume.path.mkdir(parents=True, exist_ok=True)

        progress_display = RichProgressDisplay(n or 0, queue=LocalQueue())
        # Target ~10 emissions/sec overall: interval = max_workers / target_rate_hz
        emission_interval = self.max_workers / 10.0
        local_fn = _wrap_for_local(
            fn,
            self._before_hooks,
            run_id,
            progress_display.queue,
            kwargs=kwargs or {},
            emission_interval=emission_interval,
            data_dir=self._volume.path if self._volume is not None else None,
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


def _wrap_for_local(
    fn: Callable[..., R],
    hooks: list[Callable[[], None]],
    run_id: str,
    queue: QueueLike[ProgressMessage],
    kwargs: dict[str, Any],
    emission_interval: float,
    data_dir: Path | None,
) -> Callable[..., R]:
    def run_one(index: int, *args) -> R:
        dir_ctx = data_dir_context(path=data_dir) if data_dir is not None else nullcontext()
        with progress_context(run_id, str(index), queue=queue, emission_interval=emission_interval), dir_ctx:
            for hook in reversed(hooks):
                hook()
            result = fn(*args, **kwargs)
            return result

    return run_one

"""
Progress transport for the *blocking* Modal path (``Apparatus.amap``).

A queue needs a consumer: with nobody draining it, it grows until the backend caps it. That's a fine assumption here and only here — the sole :class:`ModalQueue` in the codebase is built in :meth:`~mini.modal_apparatus.ModalApparatus._amap` over a ``modal.Queue.ephemeral()``, inside the ``async with`` that also holds its consumer (:class:`~mini.progress_display.RichProgressDisplay`). The queue is created and destroyed with that block, so it cannot outlive the reader.

The memoized orchestration path deliberately has no queue at all. Its workers are detached and its readers are short-lived polls (``status``/``watch``, a fresh process each wake), so there is no consumer to assume: progress goes straight into the control-plane ``modal.Dict``, last-writer-wins, via ``mini._taskworker._MemoSink``. See eng/operations.md.
"""

from __future__ import annotations

import logging
from collections import deque
from queue import Empty
from typing import TypeVar, cast

import modal

from mini._queues import EndOfQueue, QueueLike

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class ModalQueue(QueueLike[T]):
    """A Modal-backed queue with buffered batch reads."""

    def __init__(self, queue: modal.Queue, batch_size: int = 5_000):
        self._queue = queue
        self._batch_size = batch_size
        self._buffer: deque[T] = deque()
        self._saw_end = False

    def put(self, item: T | EndOfQueue, /, block: bool = True, timeout: float | None = None) -> None:
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, /, block: bool = True, timeout: float | None = None) -> T:
        if self._buffer:
            return self._buffer.popleft()
        if self._saw_end:
            raise EndOfQueue()

        # Modal's Queue returns None instead of raising Empty when no item is available.
        items = self._queue.get_many(self._batch_size, block=block, timeout=timeout)
        if not items:
            raise Empty("Modal queue returned no items, treating as empty")

        cleaned: list[T] = []
        for item in items:
            if isinstance(item, EndOfQueue):
                self._saw_end = True
                break
            if item is None:
                continue
            cleaned.append(cast(T, item))

        if not cleaned:
            if self._saw_end:
                raise EndOfQueue()
            raise Empty("Modal queue returned no items, treating as empty")

        self._buffer.extend(cleaned)
        return self._buffer.popleft()

    def empty(self) -> bool:
        # Modal's Queue doesn't have an empty() method.
        return self._queue.len() == 0

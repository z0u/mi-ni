"""
Apparatus for running sweeps on Modal infrastructure.

Example::

    from mini.modal_apparatus import ModalApparatus

    app = ModalApparatus("my-experiment").w(gpu="T4", timeout=3600)
    results = list(app.map(train, configs))
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

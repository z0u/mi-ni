from queue import Queue
from typing import TypeVar

from mini._queues import EndOfQueue, QueueLike

T = TypeVar("T")


class LocalQueue(QueueLike[T]):
    """A simple thread-safe queue for local use."""

    def __init__(self):
        self._queue: Queue[T | EndOfQueue] = Queue()

    def put(self, item: T | EndOfQueue, /, block: bool = True, timeout: float | None = None) -> None:
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, /, block: bool = True, timeout: float | None = None) -> T:
        item = self._queue.get(block=block, timeout=timeout)
        if isinstance(item, EndOfQueue):
            raise item
        return item

    def empty(self) -> bool:
        return self._queue.empty()

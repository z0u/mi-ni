from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


class QueueLike(Protocol, Generic[T]):
    def put(self, item: T | EndOfQueue, /, block: bool = True, timeout: float | None = None) -> None: ...
    def get(self, /, block: bool = True, timeout: float | None = None) -> T: ...
    def empty(self) -> bool: ...


class EndOfQueue(Exception):
    """A sentinel value to indicate the end of a queue."""

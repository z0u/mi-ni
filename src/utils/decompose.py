from typing import NamedTuple, Protocol, TypeVar, Union, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Gettable(Protocol):
    def get(self, key: str, default: T | None = None) -> T | None: ...


def select(obj: Gettable, keys: Union[str, list[str]]) -> tuple[T, ...]:
    """
    Extract multiple values from a gettable object using a list of keys.

    >>> config = {"database.host": "localhost", "database.port": 5432}
    >>> select(config, "database.host,database.port")
    ('localhost', 5432)
    """
    if isinstance(keys, str):
        keys = keys.replace(", ", ",").replace(" ", ",").split(",")
    return tuple(obj.get(k) for k in keys)


def selecta(obj, keys: Union[str, list[str]]) -> tuple:
    """Extract multiple fields from an object using a list of attribute names."""
    if isinstance(keys, str):
        keys = keys.replace(", ", ",").replace(" ", ",").split(",")
    return tuple(getattr(obj, k, None) for k in keys)


def sliceable(cls: NamedTuple):
    """
    Add advanced slicing to NamedTuples.

    foo['a c'], foo['b:d'], foo[[0,2]]
    """
    original_getitem = cls.__getitem__

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if ":" in idx:
                start, end = idx.split(":")
                start_idx = self._fields.index(start)
                end_idx = self._fields.index(end)
                return tuple(self[i] for i in range(start_idx, end_idx + 1))
            else:
                idx = idx.split(" ")
        if isinstance(idx, (list, tuple)):
            if all(isinstance(i, str) for i in idx):
                return tuple(getattr(self, i) for i in idx)
            return tuple(original_getitem(self, i) for i in idx)
        return original_getitem(self, idx)

    cls.__getitem__ = __getitem__
    return cls

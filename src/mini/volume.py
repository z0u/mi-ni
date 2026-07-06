"""
Storage abstraction for experiments.

Provides a portable data directory that works identically across apparatus
backends. Functions use ``get_data_dir()`` to obtain a filesystem path for
reading and writing data — the apparatus sets this up automatically via a
context variable, just like ``emit_progress()``.

Example::

    from mini.volume import get_data_dir

    def train(config):
        data_dir = get_data_dir()
        save_model(model, data_dir / 'model.pt')
"""

from __future__ import annotations

import contextvars
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Union

__all__ = ["Volume", "get_data_dir", "data_dir_context"]

PathLike = Union[str, Path, PurePosixPath]

# ---------------------------------------------------------------------------
# Context variable for data directory
# ---------------------------------------------------------------------------

_data_dir: contextvars.ContextVar[Path | None] = contextvars.ContextVar("mini_data_dir", default=None)


@contextmanager
def data_dir_context(path: Path):
    """Set the data directory for the current job context."""
    token = _data_dir.set(path)
    try:
        yield
    finally:
        _data_dir.reset(token)


def get_data_dir() -> Path:
    """
    Get the data directory for the current job.

    Must be called within an apparatus-mapped function. Raises
    ``RuntimeError`` if called outside a job context.
    """
    d = _data_dir.get()
    if d is None:
        raise RuntimeError(
            "No data directory configured. "
            "get_data_dir() must be called inside a function run by an "
            "Apparatus, and it must have a Volume configured."
        )
    return d


# ---------------------------------------------------------------------------
# Volume ABC
# ---------------------------------------------------------------------------


class Volume(ABC):
    """Abstract storage backend for an apparatus."""

    @property
    @abstractmethod
    def path(self) -> Path:
        """The filesystem path where data is stored (inside functions)."""
        ...

    @abstractmethod
    async def upload(self, local_path: PathLike, remote_path: PathLike) -> None:
        """Copy a local file or directory into the volume."""
        ...

    @abstractmethod
    async def download(self, remote_path: PathLike, local_path: PathLike) -> None:
        """Copy a file or directory from the volume to a local path."""
        ...

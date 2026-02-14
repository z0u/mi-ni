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
import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Union

import modal

__all__ = ['Volume', 'LocalVolume', 'ModalVolume', 'get_data_dir', 'data_dir_context']

PathLike = Union[str, Path, PurePosixPath]

# ---------------------------------------------------------------------------
# Context variable for data directory
# ---------------------------------------------------------------------------

_data_dir: contextvars.ContextVar[Path | None] = contextvars.ContextVar('mini_data_dir', default=None)


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
            'No data directory configured. '
            'get_data_dir() must be called inside a function run by an Apparatus.'
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
    def upload(self, local_path: PathLike, remote_path: PathLike) -> None:
        """Copy a local file or directory into the volume."""
        ...

    @abstractmethod
    def download(self, remote_path: PathLike, local_path: PathLike) -> None:
        """Copy a file or directory from the volume to a local path."""
        ...


# ---------------------------------------------------------------------------
# LocalVolume
# ---------------------------------------------------------------------------


class LocalVolume(Volume):
    """A volume backed by a local directory."""

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def upload(self, local_path: PathLike, remote_path: PathLike) -> None:
        src = Path(local_path)
        dst = self._path / remote_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def download(self, remote_path: PathLike, local_path: PathLike) -> None:
        src = self._path / remote_path
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# ModalVolume
# ---------------------------------------------------------------------------


class ModalVolume(Volume):
    """A volume backed by a Modal named volume."""

    def __init__(self, name: str, mount_point: str = '/vol'):
        self._name = name
        self._mount_point = Path(mount_point)
        self._modal_volume = modal.Volume.from_name(name, create_if_missing=True)

    @property
    def path(self) -> Path:
        return self._mount_point

    def upload(self, local_path: PathLike, remote_path: PathLike) -> None:
        src = Path(local_path)
        with self._modal_volume.batch_upload() as batch:
            if src.is_dir():
                batch.put_directory(str(src), str(remote_path))
            else:
                batch.put_file(str(src), str(remote_path))

    def download(self, remote_path: PathLike, local_path: PathLike) -> None:
        remote = PurePosixPath(remote_path)
        dst = Path(local_path)

        # Try reading as a single file first
        entries = list(self._modal_volume.listdir(str(remote)))
        if len(entries) == 1 and entries[0].path == str(remote):
            # It's a single file
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, 'wb') as f:
                for chunk in self._modal_volume.read_file(str(remote)):
                    f.write(chunk)
        else:
            # It's a directory — download each entry
            dst.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                entry_remote = PurePosixPath(entry.path)
                entry_local = dst / entry_remote.relative_to(remote)
                if entry.type.name == 'FILE':
                    entry_local.parent.mkdir(parents=True, exist_ok=True)
                    with open(entry_local, 'wb') as f:
                        for chunk in self._modal_volume.read_file(entry.path):
                            f.write(chunk)

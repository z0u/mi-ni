"""
Volume backed by a local directory.
"""

from __future__ import annotations
from typing import override

import shutil
from pathlib import Path

from mini.volume import PathLike, Volume

__all__ = ["LocalVolume"]


class LocalVolume(Volume):
    """
    A volume backed by a local directory.

    The directory is created lazily (the first time data is read or written).
    """

    def __init__(self, path: Path | str):
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    @override
    async def upload(self, local_path: PathLike, remote_path: PathLike) -> None:
        """
        Copy a local file or directory into the volume.

        ``remote_path`` is the **full destination path** within the volume,
        not a parent directory. Parent directories are created automatically.

        For files::

            vol.upload('results/scores.csv', 'output/scores.csv')
            # → <vol>/output/scores.csv

        For directories::

            vol.upload('results/run-1', 'output/run-1')
            # → <vol>/output/run-1/{contents of results/run-1/}

        If the destination directory already exists, its contents are merged
        rather than replaced.
        """
        src = Path(local_path)
        dst = self._path / remote_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    @override
    async def download(self, remote_path: PathLike, local_path: PathLike) -> None:
        """
        Copy a file or directory from the volume to a local path.

        ``local_path`` is the **full destination path**, not a parent directory.
        Parent directories are created automatically.

        For files::

            vol.download('output/scores.csv', '/tmp/scores.csv')
            # → /tmp/scores.csv

        For directories::

            vol.download('output/run-1', '/tmp/run-1')
            # → /tmp/run-1/{contents of <vol>/output/run-1/}

        If the destination directory already exists, its contents are merged
        rather than replaced.
        """
        src = self._path / remote_path
        dst = Path(local_path)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

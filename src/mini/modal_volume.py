"""
Volume backed by a Modal named volume.
"""

from __future__ import annotations
from typing import override

from pathlib import Path, PurePosixPath

import modal

from mini.volume import PathLike, Volume

__all__ = ["ModalVolume"]


class ModalVolume(Volume):
    """
    A volume backed by a Modal named volume.

    The volume is mounted at ``mount_point`` inside Modal functions. The
    mount point is created by Modal automatically when the volume is attached.
    """

    def __init__(self, name: str, mount_point: str = "/vol", *, create: bool = True):
        # ``create=False`` for read-only peeks (gc's mark phase): a look at an
        # experiment's volume must not mint an empty one on Modal.
        self._mount_point = Path(mount_point)
        self._modal_volume = modal.Volume.from_name(name, create_if_missing=create)

    @property
    def path(self) -> Path:
        return self._mount_point

    @override
    async def upload(self, local_path: PathLike, remote_path: PathLike) -> None:
        """
        Copy a local file or directory into the volume.

        ``remote_path`` is the **full destination path** within the volume,
        not a parent directory.

        For files::

            vol.upload('results/scores.csv', 'output/scores.csv')
            # → <vol>/output/scores.csv

        For directories::

            vol.upload('results/run-1', 'output/run-1')
            # → <vol>/output/run-1/{contents of results/run-1/}
        """
        src = Path(local_path)
        async with self._modal_volume.batch_upload.aio() as batch:
            if src.is_dir():
                batch.put_directory(str(src), str(remote_path))
            else:
                batch.put_file(str(src), str(remote_path))

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
        """
        remote = PurePosixPath(remote_path)
        dst = Path(local_path)

        entries = list(await self._modal_volume.listdir.aio(str(remote)))
        if len(entries) == 1 and entries[0].path == str(remote):
            # Single file
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                async for chunk in self._modal_volume.read_file.aio(str(remote)):
                    f.write(chunk)
        else:
            # Directory — download each entry relative to remote root
            dst.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                entry_remote = PurePosixPath(entry.path)
                entry_local = dst / entry_remote.relative_to(remote)
                if entry.type.name == "FILE":
                    entry_local.parent.mkdir(parents=True, exist_ok=True)
                    with open(entry_local, "wb") as f:
                        async for chunk in self._modal_volume.read_file.aio(entry.path):
                            f.write(chunk)

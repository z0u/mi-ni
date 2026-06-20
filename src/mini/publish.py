"""
Publish finished experiment artifacts to the Hugging Face Hub.

A :class:`~mini.volume.Volume` is *compute-attached* storage: mounted, private,
and reachable only through the apparatus' SDK. The Hub is the opposite — a
public, versioned, URL-addressable sink for *finished* artifacts: datasets,
model weights, and the large media that notebooks embed. Files are served over a
CDN at a stable ``resolve`` URL, so a published notebook can point an ``<img>``
or ``<video>`` straight at the Hub instead of bloating Git LFS.

Auth is a token (``HF_TOKEN``), so this works from anywhere — including cloud
agents that can't write Git LFS.

Example::

    from mini import HFStore

    store = HFStore('z0u/mi-ni-artifacts')           # a dataset repo
    url = store.publish('out/loss.mp4', 'gpt/loss.mp4')
    mo.Html(f'<video src="{url}" controls></video>')  # served from the CDN
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Literal

from mini.volume import PathLike

__all__ = ['HFStore', 'RepoType']

RepoType = Literal['dataset', 'model', 'space']

# Where ``resolve`` URLs live, per repo type. Models sit at the root; datasets
# and spaces are namespaced under a path segment.
_URL_PREFIX: dict[RepoType, str] = {
    'model': 'https://huggingface.co/{repo_id}/resolve/{rev}/{path}',
    'dataset': 'https://huggingface.co/datasets/{repo_id}/resolve/{rev}/{path}',
    'space': 'https://huggingface.co/spaces/{repo_id}/resolve/{rev}/{path}',
}


class HFStore:
    """
    A publishing sink backed by a Hugging Face Hub repository.

    Unlike a :class:`~mini.volume.Volume`, this is not mounted during compute and
    has no ``path`` inside functions — it's a place to *send* artifacts once
    they're done, and to *link* them from published notebooks.

    Args:
        repo_id: ``'owner/name'`` of the Hub repository.
        repo_type: ``'dataset'`` (default), ``'model'`` or ``'space'``.
        revision: Branch or tag to write to and link against. Pin a tag for
            reproducible report URLs.
        token: Hub token. Defaults to ``$HF_TOKEN`` (then ``$HUGGINGFACE_TOKEN``).
        private: Create the repo as private if it doesn't exist yet.
        create: Create the repo (if missing) on first write.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        repo_type: RepoType = 'dataset',
        revision: str = 'main',
        token: str | None = None,
        private: bool = False,
        create: bool = True,
    ):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision
        self._private = private
        self._create = create
        self._created = False
        self._token = token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        self._api = HfApi(token=self._token)

    def url(self, remote_path: PathLike, *, revision: str | None = None) -> str:
        """
        The public ``resolve`` URL for a path in the repo.

        This is what you embed in a notebook (``<img src=…>``) — it works without
        a token for public repos and is served from the CDN.
        """
        return _URL_PREFIX[self.repo_type].format(
            repo_id=self.repo_id,
            rev=revision or self.revision,
            path=str(PurePosixPath(remote_path)),
        )

    def publish(
        self,
        local_path: PathLike,
        remote_path: PathLike | None = None,
        *,
        commit_message: str | None = None,
    ) -> str:
        """
        Upload a file or directory to the repo and return its public URL.

        ``remote_path`` is the destination path within the repo; it defaults to
        the source's basename. For a directory, the returned URL points at the
        destination folder.

        ::

            store.publish('out/model.eqx', 'nanogpt/model.eqx')   # one file
            store.publish('out/run-1', 'nanogpt/run-1')           # a tree
        """
        src = Path(local_path)
        dest = PurePosixPath(remote_path) if remote_path is not None else PurePosixPath(src.name)
        self._ensure_repo()

        if src.is_dir():
            (
                self._api.upload_folder(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=self.revision,
                    folder_path=str(src),
                    path_in_repo=str(dest),
                    commit_message=commit_message or f'Publish {dest}/',
                )
            )
        else:
            (
                self._api.upload_file(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=self.revision,
                    path_or_fileobj=str(src),
                    path_in_repo=str(dest),
                    commit_message=commit_message or f'Publish {dest}',
                )
            )
        return self.url(dest)

    def fetch(self, remote_path: PathLike, local_path: PathLike) -> None:
        """
        Download a file or directory from the repo to ``local_path``.

        The complement of :meth:`publish`, for reports that prefer to pull bytes
        at execution time rather than link the CDN. ``local_path`` is the full
        destination path; parent directories are created as needed.
        """
        import shutil

        from huggingface_hub import hf_hub_download, snapshot_download

        remote = PurePosixPath(remote_path)
        dst = Path(local_path)

        # A trailing slash, or no file suffix, is treated as a directory subtree.
        is_dir = str(remote_path).endswith('/') or not remote.suffix
        if is_dir:
            snap = snapshot_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=self.revision,
                token=self._token,
                allow_patterns=[f'{remote}/*'],
            )
            shutil.copytree(Path(snap) / remote, dst, dirs_exist_ok=True)
        else:
            cached = hf_hub_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=self.revision,
                token=self._token,
                filename=str(remote),
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, dst)

    def _ensure_repo(self) -> None:
        if self._created or not self._create:
            return
        self._api.create_repo(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            private=self._private,
            exist_ok=True,
        )
        self._created = True

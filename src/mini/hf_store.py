"""
Hugging Face bucket backend for the artifact :class:`~mini.store.Store`.

A bucket (``hf://buckets/<namespace>/<name>``) is a Xet-backed, mutable repo with
no git history — so concurrent writers don't conflict and immutability is ours to
enforce by writing once per content hash. We lay the same ``cas/<sha256>`` /
``refs/`` / ``published/`` structure over it as :class:`~mini.store.LocalStore`,
so an :class:`~mini.store.Artifact` handle resolves identically whichever backend
produced it.

Three properties make this the durable, shareable tier:

- **One bucket per project** → ``has(sha)`` is a cross-experiment hit, so a blob
  one experiment uploads is skipped (not re-uploaded) by another, and Xet dedups
  the chunks underneath for free.
- **Reachable everywhere** — from a Modal worker, a local report, or a browser —
  so it retires the per-experiment-Volume limitation for *artifacts* (the Volume
  becomes an optional warm cache, not the source of truth).
- **Web-serving for free** — :meth:`publish` server-side-copies a blob *by xet
  hash* (no bytes moved) to an extensioned path, and the bucket's resolve URL
  then serves it with a ``Content-Type`` inferred from that extension.

Blobs are warm-cached into a local :class:`~mini.store.LocalStore` so a re-read
(or a re-``put`` of known bytes) skips the network. The bucket stays the source
of truth; the cache is just an accelerator.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from mini.store import Artifact, LocalStore, Store, _cas_key, _hash_file, _tree_sha

__all__ = ['HFStore']

# Buckets need ``*.xethub.hf.co`` (byte transfer) and, for serving, ``*.cdn.hf.co``
# on the network egress allow-list; metadata-only calls to ``huggingface.co`` work
# without them but every transfer hangs on a 403. See research/hf-buckets.md.


class HFStore(Store):
    """A :class:`~mini.store.Store` backed by a Hugging Face bucket via ``HfApi``."""

    def __init__(self, bucket: str, *, cache: LocalStore, token: str | None = None):
        self.bucket = bucket
        self._cache = cache  # local warm checkout, keyed by sha (a LocalStore)
        self._token = token or os.environ.get('HF_TOKEN')
        self._api: Any = None

    @property
    def api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self._token)
        return self._api

    # -- existence / cache ----------------------------------------------------

    def _remote_has(self, path: str) -> bool:
        # A missing path is "absent" (return False); an auth/permission/network
        # failure must *not* masquerade as absent — that would silently trigger a
        # re-upload (and hide a misconfigured token), so let those propagate.
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

        try:
            return any(True for _ in self.api.get_bucket_paths_info(self.bucket, [path]))
        except EntryNotFoundError, RepositoryNotFoundError:
            return False

    def has(self, sha256: str) -> bool:
        return self._cache.has(sha256) or self._remote_has(_cas_key(sha256))

    def is_published(self, path: str) -> bool:
        """Whether a ``published/<path>`` view already exists (a read).

        Lets a publish step short-circuit: the published asset paths are
        content-addressed (``…/_assets/<sha>/<name>``), so an existing path is the
        same bytes. A build with a **read-only** token (CI) can thus skip the
        server-side copy when the agent already published the assets at export time.
        """
        return self._remote_has(f'published/{path}')

    def _cache_blob(self, sha256: str, src: Path) -> None:
        if not self._cache.has(sha256):
            self._cache._write_blob(sha256, src)

    # -- blobs ----------------------------------------------------------------

    def _write_blob(self, sha256: str, src: Path) -> None:
        # Reached only on a cache+remote miss (the base ``put`` checks ``has``
        # first); Xet still dedups the chunks if the bytes happen to exist.
        self.api.batch_bucket_files(self.bucket, add=[(str(src), _cas_key(sha256))])
        self._cache_blob(sha256, src)

    def _read_blob(self, sha256: str, dest: Path) -> None:
        blob = self._cache._blob_path(sha256)
        if not blob.exists():  # pull once into the warm cache, then serve locally
            blob.parent.mkdir(parents=True, exist_ok=True)
            self.api.download_bucket_files(self.bucket, files=[(_cas_key(sha256), str(blob))])
        shutil.copyfile(blob, dest)

    def _put_tree(self, src: Path, *, name: str) -> Artifact:
        """Hash every shard locally, then upload the missing ones in **one** commit.

        Batching matters here: each bucket commit pays a ~2-3s round trip, so a
        per-shard upload would serialize that floor across the whole tree.
        """
        children: list[Artifact] = []
        add: list[tuple[str, str]] = []
        for p in sorted(q for q in src.rglob('*') if q.is_file()):
            sha, size = _hash_file(p)
            children.append(Artifact(sha256=sha, size=size, name=p.relative_to(src).as_posix()))
            if not self._cache.has(sha):
                add.append((str(p), _cas_key(sha)))
            self._cache_blob(sha, p)
        if add:
            self.api.batch_bucket_files(self.bucket, add=add)  # one round trip for the set
        kids = tuple(children)
        return Artifact(sha256=_tree_sha(kids), size=sum(c.size for c in kids), name=name, kind='tree', children=kids)

    # -- refs -----------------------------------------------------------------

    def _write_ref(self, name: str, payload: str) -> None:
        self.api.batch_bucket_files(self.bucket, add=[(payload.encode(), f'refs/{name}.json')])

    def _read_ref(self, name: str) -> str | None:
        path = f'refs/{name}.json'
        if not self._remote_has(path):
            return None
        with tempfile.TemporaryDirectory() as d:  # cleaned up, unlike a bare mkdtemp
            tmp = Path(d) / 'ref.json'
            self.api.download_bucket_files(self.bucket, files=[(path, str(tmp))])
            return tmp.read_text()

    # -- publish --------------------------------------------------------------

    def publish(self, art: Artifact, path: str) -> str:
        if art.kind == 'tree':
            raise ValueError('publish a single file (resolve a tree first, or publish its children)')
        info = list(self.api.get_bucket_paths_info(self.bucket, [_cas_key(art.sha256)]))
        if not info:
            raise FileNotFoundError(f'{art.sha256[:12]}… is not in the store — put() it before publish()')
        # Server-side copy *by xet hash*: a metadata op, no bytes moved. The
        # extensioned destination is what makes the resolve URL serve a real
        # Content-Type (a bare cas/<sha> has none).
        dest = f'published/{path}'
        self.api.batch_bucket_files(self.bucket, copy=[('bucket', self.bucket, info[0].xet_hash, dest)])
        return f'https://huggingface.co/buckets/{self.bucket}/resolve/{dest}'

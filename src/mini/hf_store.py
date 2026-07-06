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

**The publish tier can live in a separate dataset repo.** ``put``/``get``/refs run
hot and concurrent (many workers), which is why the CAS is a bucket — buckets have
no git history, so parallel writers never conflict. ``publish`` and report exports
run cold and single-writer (a driver or CI), so they can afford a git-backed
Hugging Face *dataset repo*, which buys two things a bucket can't: a **public**
face over a **private** CAS (buckets have no per-prefix ACL, so this is a genuine
two-store split), and **versioned names** — a citation pins to a commit sha. Set
``publish_repo`` (``[tool.mini] publish-repo``) to route publish/exports there;
unset, they stay in the bucket. See ``eng/publishing.md`` and issue #38.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from mini.store import Artifact, BlobStat, LocalStore, Store, _cas_key, _hash_file, _tree_sha

__all__ = ["HFStore"]

# Buckets need ``*.xethub.hf.co`` (byte transfer) and, for serving, ``*.cdn.hf.co``
# on the network egress allow-list; metadata-only calls to ``huggingface.co`` work
# without them but every transfer hangs on a 403. See eng/operations.md.


class HFStore(Store):
    """A :class:`~mini.store.Store` backed by a Hugging Face bucket via ``HfApi``."""

    def __init__(
        self,
        bucket: str | None,
        *,
        cache: LocalStore,
        token: str | None = None,
        publish_repo: str | None = None,
    ):
        # ``None`` is the publish-only store: no CAS bucket, just the dataset repo
        # for reading/serving exports (the read-only site build, which never touches
        # the CAS). CAS/ref operations raise via :attr:`_cas`. A publish_repo must be
        # set in that case — ``store_for`` only builds this with a bucket *or* a repo.
        self.bucket = bucket
        # The public, versioned publish tier: a Hugging Face *dataset* repo (real git
        # history → a citation pins to a commit sha) that backs publish() and report
        # exports. ``None`` keeps both in ``bucket`` — the single-store default, which
        # is also what makes the split opt-in and this class backend-swappable (#38).
        self.publish_repo = publish_repo
        self._cache = cache  # local warm checkout, keyed by sha (a LocalStore)
        self._token = token or os.environ.get("HF_TOKEN")
        self._api: Any = None

    @property
    def api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self._token)
        return self._api

    @property
    def _cas(self) -> str:
        """The bucket backing the CAS/refs — or a clear error if this store is publish-only.

        A store built from a publish-repo alone (no ``store-bucket``) can serve exports
        but has nowhere to put/get blobs, so every CAS/ref path narrows through here.
        """
        if self.bucket is None:
            raise RuntimeError(
                "this store has no CAS bucket — it was built from a publish-repo alone "
                "(read-only export serving). Set store-bucket (or MINI_STORE_BUCKET) to "
                "put/get/ref against the content-addressed store."
            )
        return self.bucket

    # -- existence / cache ----------------------------------------------------

    def _remote_has(self, path: str) -> bool:
        # A missing path is "absent" (return False); an auth/permission/network
        # failure must *not* masquerade as absent — that would silently trigger a
        # re-upload (and hide a misconfigured token), so let those propagate.
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

        try:
            return any(True for _ in self.api.get_bucket_paths_info(self._cas, [path]))
        except EntryNotFoundError, RepositoryNotFoundError:
            return False

    def has(self, sha256: str) -> bool:
        return self._cache.has(sha256) or self._remote_has(_cas_key(sha256))

    def _cache_blob(self, sha256: str, src: Path) -> None:
        if not self._cache.has(sha256):
            self._cache._write_blob(sha256, src)

    # -- blobs ----------------------------------------------------------------

    def _write_blob(self, sha256: str, src: Path) -> None:
        # Reached only on a cache+remote miss (the base ``put`` checks ``has``
        # first); Xet still dedups the chunks if the bytes happen to exist.
        self.api.batch_bucket_files(self._cas, add=[(str(src), _cas_key(sha256))])
        self._cache_blob(sha256, src)

    def _local_blob(self, sha256: str) -> Path:
        """The blob's path in the warm cache, pulled once from the CAS bucket if absent.

        The single place bytes come off the bucket: :meth:`_read_blob` serves reads
        from it, and :meth:`publish` needs a local file to hand the (separate) publish
        repo so Xet can chunk it — the durable copy lives in the CAS, not the publish
        tier, so publishing pulls-then-uploads rather than moving bytes server-side.
        """
        blob = self._cache._blob_path(sha256)
        if not blob.exists():  # pull once into the warm cache, then serve locally
            blob.parent.mkdir(parents=True, exist_ok=True)
            self.api.download_bucket_files(self._cas, files=[(_cas_key(sha256), str(blob))])
        return blob

    def _read_blob(self, sha256: str, dest: Path) -> None:
        shutil.copyfile(self._local_blob(sha256), dest)

    def _put_tree(self, src: Path, *, name: str) -> Artifact:
        """Hash every shard locally, then upload the missing ones in **one** commit.

        Batching matters here: each bucket commit pays a ~2-3s round trip, so a
        per-shard upload would serialize that floor across the whole tree.
        """
        children: list[Artifact] = []
        add: list[tuple[str, str]] = []
        for p in sorted(q for q in src.rglob("*") if q.is_file()):
            sha, size = _hash_file(p)
            children.append(Artifact(sha256=sha, size=size, name=p.relative_to(src).as_posix()))
            if not self._cache.has(sha):
                add.append((str(p), _cas_key(sha)))
            self._cache_blob(sha, p)
        if add:
            self.api.batch_bucket_files(self._cas, add=add)  # one round trip for the set
        kids = tuple(children)
        return Artifact(sha256=_tree_sha(kids), size=sum(c.size for c in kids), name=name, kind="tree", children=kids)

    # -- refs -----------------------------------------------------------------

    def _write_ref(self, name: str, payload: str) -> None:
        self.api.batch_bucket_files(self._cas, add=[(payload.encode(), f"refs/{name}.json")])

    def _read_ref(self, name: str) -> str | None:
        path = f"refs/{name}.json"
        if not self._remote_has(path):
            return None
        with tempfile.TemporaryDirectory() as d:  # cleaned up, unlike a bare mkdtemp
            tmp = Path(d) / "ref.json"
            self.api.download_bucket_files(self._cas, files=[(path, str(tmp))])
            return tmp.read_text()

    # -- publish --------------------------------------------------------------

    def publish(self, art: Artifact, path: str) -> str:
        if art.kind == "tree":
            raise ValueError("publish a single file (resolve a tree first, or publish its children)")
        if self.publish_repo is not None:
            return self._publish_to_repo(art, path)
        return self._publish_to_bucket(art, path)

    def _publish_to_bucket(self, art: Artifact, path: str) -> str:
        """The single-store default: expose a CAS blob in-bucket under ``published/``."""
        bucket = self._cas
        info = list(self.api.get_bucket_paths_info(bucket, [_cas_key(art.sha256)]))
        if not info:
            raise FileNotFoundError(f"{art.sha256[:12]}… is not in the store — put() it before publish()")
        # Server-side copy *by xet hash*: a metadata op, no bytes moved. The
        # extensioned destination is what makes the resolve URL serve a real
        # Content-Type (a bare cas/<sha> has none).
        dest = f"published/{path}"
        self.api.batch_bucket_files(bucket, copy=[("bucket", bucket, info[0].xet_hash, dest)])
        return f"https://huggingface.co/buckets/{bucket}/resolve/{dest}"

    def _publish_to_repo(self, art: Artifact, path: str) -> str:
        """Expose a CAS blob on the public, versioned dataset repo (see :func:`publish_repo`).

        No server-side by-hash copy exists across a bucket→repo boundary, so this
        pulls the blob from the (possibly private) CAS into the warm cache and uploads
        it. That's not a byte re-transfer: Xet chunk dedup is account-wide, so chunks
        the CAS already stored aren't sent again — the upload is a metadata + git-commit
        op. The commit is what gives the publish tier history: the returned ``resolve``
        URL tracks the branch, and a citation can pin the same path to a commit sha.
        """
        if not self.has(art.sha256):
            raise FileNotFoundError(f"{art.sha256[:12]}… is not in the store — put() it before publish()")
        dest = f"published/{path}"
        self.api.upload_file(
            path_or_fileobj=str(self._local_blob(art.sha256)),
            path_in_repo=dest,
            repo_id=self.publish_repo,
            repo_type="dataset",
            commit_message=f"publish {path}",
        )
        return f"https://huggingface.co/datasets/{self.publish_repo}/resolve/main/{dest}"

    # -- gc --------------------------------------------------------------------

    def _list_tree(self, prefix: str) -> Iterator[Any]:
        # A prefix that has never been written is "nothing there", not an error;
        # anything else (auth, missing bucket) must propagate — see _remote_has.
        from huggingface_hub.errors import EntryNotFoundError

        try:
            yield from self.api.list_bucket_tree(self._cas, prefix=prefix, recursive=True)
        except EntryNotFoundError:
            return

    def list_blobs(self) -> Iterator[BlobStat]:
        for entry in self._list_tree("cas"):
            if getattr(entry, "type", None) != "file":
                continue
            sha = entry.path.rsplit("/", 1)[-1]
            if len(sha) != 64 or not set(sha) <= set("0123456789abcdef"):
                continue
            ts = entry.uploaded_at or entry.mtime
            yield BlobStat(sha, entry.size, ts.timestamp() if ts is not None else None)

    def delete_blobs(self, sha256s: Iterable[str]) -> None:
        shas = list(sha256s)
        for i in range(0, len(shas), 500):  # one commit per chunk, not per blob
            self.api.batch_bucket_files(self._cas, delete=[_cas_key(s) for s in shas[i : i + 500]])
        # Purge the warm cache too: a stale local copy would make ``has`` claim
        # the bucket still holds bytes it no longer does, and a later ``put`` of
        # the same content would silently skip the re-upload.
        for s in shas:
            self._cache._blob_path(s).unlink(missing_ok=True)

    def list_refs(self) -> list[str]:
        return sorted(
            e.path[len("refs/") : -len(".json")]
            for e in self._list_tree("refs")
            if getattr(e, "type", None) == "file" and e.path.endswith(".json")
        )

    # -- report bundles (the publish-a-report handoff) ------------------------
    #
    # A report is exported (HTML + named-keyed _assets/) to a self-contained local dir,
    # then mirrored *as-is* to ``exports/<key>/``. The build (CI) reads those back and
    # assembles the site, pointing a single <base> at ``exports/<key>/``. This keeps the
    # heavy/authenticated half (export, which needs the data + a write token) on the
    # agent, and the deterministic half (link resolution, <base>) read-only in CI — no
    # cas/<sha>, no publish() copy, no accumulation (names overwrite in place).

    def export_base(self, key: str) -> str:
        """The ``<base href>`` a published report's relative ``_assets/`` resolve against."""
        if self.publish_repo is not None:
            return f"https://huggingface.co/datasets/{self.publish_repo}/resolve/main/exports/{key}/"
        return f"https://huggingface.co/buckets/{self.bucket}/resolve/exports/{key}/"

    def sync_export(self, local_dir: Path, key: str) -> None:
        """Mirror a report's local export dir to ``exports/<key>/`` (delete stale).

        rsync-like — a re-export overwrites in place and drops assets the report no
        longer references — with the per-report bundle as the sync unit. On the bucket
        that's ``sync_bucket``; on the dataset repo it's one commit whose
        ``delete_patterns`` prunes the bundle's now-absent files.
        """
        if self.publish_repo is not None:
            self.api.upload_folder(
                folder_path=str(local_dir),
                path_in_repo=f"exports/{key}",
                repo_id=self.publish_repo,
                repo_type="dataset",
                delete_patterns="*",
                commit_message=f"export {key}",
            )
            return
        self.api.sync_bucket(source=str(local_dir), dest=f"hf://buckets/{self.bucket}/exports/{key}", delete=True)

    def fetch_export(self, key: str, dest: Path) -> bool:
        """Download ``exports/<key>/`` into *dest*; ``False`` if nothing is synced.

        The build reads exports back this way — read-only, no notebook execution — so a
        report missing here just means it hasn't been ``./go publish``ed yet.
        """
        if self.publish_repo is not None:
            return self._fetch_export_from_repo(key, dest)
        if not self._remote_has(f"exports/{key}/index.html"):
            return False
        dest.mkdir(parents=True, exist_ok=True)
        self.api.sync_bucket(source=f"hf://buckets/{self.bucket}/exports/{key}", dest=str(dest))
        return True

    def _fetch_export_from_repo(self, key: str, dest: Path) -> bool:
        from huggingface_hub import snapshot_download

        repo = self.publish_repo
        assert repo is not None  # only reached from fetch_export when the publish tier is a repo
        prefix = f"exports/{key}"
        if not self.api.file_exists(repo_id=repo, filename=f"{prefix}/index.html", repo_type="dataset"):
            return False
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            snap = snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=f"{prefix}/*",
                local_dir=d,
                token=self._token,
            )
            src = Path(snap) / prefix
            for p in src.rglob("*"):  # lift the bundle out of its exports/<key>/ prefix into dest
                if p.is_file():
                    out = dest / p.relative_to(src)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, out)
        return True

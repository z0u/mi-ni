"""
Hugging Face bucket backend for the artifact :class:`~mini.store.Store`.

A bucket (``hf://buckets/<namespace>/<name>``) is a Xet-backed, mutable repo with no git history — so concurrent writers don't conflict and immutability is ours to enforce by writing once per content hash. We lay the same ``cas/<sha256>`` / ``refs/`` / ``published/`` structure over it as :class:`~mini.store.LocalStore`, so an :class:`~mini.store.Artifact` handle resolves identically whichever backend produced it.

Three properties make this the durable, shareable tier:

- **One bucket per project** → ``has(sha)`` is a cross-experiment hit, so a blob one experiment uploads is skipped (not re-uploaded) by another, and Xet dedups the chunks underneath for free.
- **Reachable everywhere** — from a Modal worker, a local report, or a browser — so it retires the per-experiment-Volume limitation for *artifacts* (the Volume becomes an optional warm cache, not the source of truth).
- **Web-serving for free** — :meth:`publish` server-side-copies a blob *by xet hash* (no bytes moved) to an extensioned path, and the bucket's resolve URL then serves it with a ``Content-Type`` inferred from that extension.

Blobs are warm-cached into a local :class:`~mini.store.LocalStore` so a re-read (or a re-``put`` of known bytes) skips the network. The bucket stays the source of truth; the cache is just an accelerator.

**The publish tier can live in a separate dataset repo.** ``put``/``get``/refs run hot and concurrent (many workers), which is why the CAS is a bucket — buckets have no git history, so parallel writers never conflict. ``publish`` and report exports run cold and single-writer (a driver or CI), so they can afford a git-backed Hugging Face *dataset repo*, which buys two things a bucket can't: a **public** face over a **private** CAS (buckets have no per-prefix ACL, so this is a genuine two-store split), and **versioned names** — a citation pins to a commit sha. Set ``publish_repo`` (``[tool.mini] publish-repo``) to route publish/exports there; unset, they stay in the bucket. See ``eng/publishing.md`` and issue #38.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

from mini.store import Artifact, BlobStat, LocalStore, Store, _cas_key, _hash_file, _tree_sha, artifact_shas

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
        self._api_lock = threading.Lock()

    @property
    def api(self) -> Any:
        # Locked because a store is read from several threads at once (the site build
        # fetches its reports in one wave), and one shared HfApi is the point.
        with self._api_lock:
            if self._api is None:
                from huggingface_hub import HfApi

                self._api = HfApi(token=self._token)
            return self._api

    @property
    def _cas(self) -> str:
        """The bucket backing the CAS/refs — or a clear error if this store is publish-only.

        A store built from a publish-repo alone (no ``store-bucket``) can serve exports but has nowhere to put/get blobs, so every CAS/ref path narrows through here.
        """
        if self.bucket is None:
            raise RuntimeError(
                "this store has no CAS bucket — it was built from a publish-repo alone "
                "(read-only export serving). Set store-bucket (or MINI_STORE_BUCKET) to "
                "put/get/ref against the content-addressed store."
            )
        return self.bucket

    # -- existence / cache ----------------------------------------------------

    def _paths_info(self, paths: list[str]) -> dict[str, Any]:
        """Path → ``BucketFile`` for the *paths* that exist, in one batched request.

        A missing path is simply absent from the result; an auth/permission/network failure must *not* masquerade as absent — that would silently trigger a re-upload (and hide a misconfigured token), so only the two not-found errors are caught and everything else propagates.
        """
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

        if not paths:
            return {}
        try:
            return {info.path: info for info in self.api.get_bucket_paths_info(self._cas, paths)}
        except EntryNotFoundError, RepositoryNotFoundError:
            return {}

    def _remote_has(self, path: str) -> bool:
        return bool(self._paths_info([path]))

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

    def _pull_blobs(self, sha256s: Iterable[str]) -> None:
        """Pull every warm-cache-missing blob off the CAS bucket in **one** request.

        The single place bytes come off the bucket. Batching matters as much on reads as on writes: each ``download_bucket_files`` call pays the fixed paths-info + metadata round trips before any bytes move, so pulling *n* blobs one call at a time serializes that floor *n* times over.
        """
        missing = [s for s in dict.fromkeys(sha256s) if not self._cache.has(s)]
        if not missing:
            return
        infos = self._paths_info([_cas_key(s) for s in missing])
        if absent := [s for s in missing if _cas_key(s) not in infos]:
            raise FileNotFoundError(f"not in the store bucket: {', '.join(s[:12] + '…' for s in absent)}")
        # Download to sibling temp files, then atomically rename in: an interrupted
        # download must never leave a partial/0-byte file at the blob path, since the
        # only cache-hit check is ``blob.exists()`` — a truncated file would then be
        # served forever (and fail to parse downstream) rather than re-pulled.
        pulls = []
        for sha in missing:
            blob = self._cache._blob_path(sha)
            blob.parent.mkdir(parents=True, exist_ok=True)
            pulls.append((blob, blob.with_name(f"{sha}.tmp.{os.getpid()}.{threading.get_ident()}")))
        try:
            self.api.download_bucket_files(
                self._cas, files=[(infos[_cas_key(s)], str(tmp)) for s, (_, tmp) in zip(missing, pulls, strict=True)]
            )
            for blob, tmp in pulls:
                tmp.replace(blob)
        finally:
            for _, tmp in pulls:
                tmp.unlink(missing_ok=True)

    def _prefetch(self, arts: Iterable[Artifact]) -> None:
        self._pull_blobs(artifact_shas(list(arts)))

    def _local_blob(self, sha256: str) -> Path:
        """The blob's path in the warm cache, pulled once from the CAS bucket if absent.

        :meth:`_read_blob` serves reads from it, and :meth:`publish` needs a local file to hand the (separate) publish repo so Xet can chunk it — the durable copy lives in the CAS, not the publish tier, so publishing pulls-then-uploads rather than moving bytes server-side.
        """
        self._pull_blobs([sha256])
        return self._cache._blob_path(sha256)

    def _read_blob(self, sha256: str, dest: Path) -> None:
        shutil.copyfile(self._local_blob(sha256), dest)

    def _put_tree(self, src: Path, *, name: str) -> Artifact:
        """Hash every shard locally, then upload the missing ones in **one** commit.

        Batching matters here: each bucket commit pays a ~2-3s round trip, so a per-shard upload would serialize that floor across the whole tree.
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
        return self._read_refs([name])[name]

    def _read_refs(self, names: list[str]) -> dict[str, str | None]:
        # One paths-info request tells present from absent (no per-name existence
        # probe), and the present ones download in a single batched call — so a
        # report resolving a dozen refs pays the bucket's round-trip floor once.
        paths = {f"refs/{n}.json": n for n in names}
        infos = self._paths_info(list(paths))
        out: dict[str, str | None] = dict.fromkeys(names)
        if not infos:
            return out
        with tempfile.TemporaryDirectory() as d:  # cleaned up, unlike a bare mkdtemp
            files = [(info, str(Path(d) / f"{i}.json")) for i, info in enumerate(infos.values())]
            self.api.download_bucket_files(self._cas, files=files)
            for path, (_, tmp) in zip(infos, files, strict=True):
                out[paths[path]] = Path(tmp).read_text()
        return out

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

        No server-side by-hash copy exists across a bucket→repo boundary, so this pulls the blob from the (possibly private) CAS into the warm cache and uploads it. That's not a byte re-transfer: Xet chunk dedup is account-wide, so chunks the CAS already stored aren't sent again — the upload is a metadata + git-commit op. The commit is what gives the publish tier history: the returned ``resolve`` URL tracks the branch, and a citation can pin the same path to a commit sha.
        """
        if not self.has(art.sha256):
            raise FileNotFoundError(f"{art.sha256[:12]}… is not in the store — put() it before publish()")
        dest = f"published/{path}"
        info = self.api.upload_file(
            path_or_fileobj=str(self._local_blob(art.sha256)),
            path_in_repo=dest,
            repo_id=self.publish_repo,
            repo_type="dataset",
            commit_message=f"publish {path}",
        )
        # Pin the returned URL to the commit, not the branch: the whole point of a
        # git-backed publish tier is that a published URL can't be swapped out from
        # under whoever holds it. (An identical re-publish creates no new commit —
        # huggingface_hub drops no-op operations and returns the current head.)
        return f"https://huggingface.co/datasets/{self.publish_repo}/resolve/{info.oid}/{dest}"

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
    # then mirrored *as-is* to ``exports/<key>/``. The build (CI) reads back the one file
    # it rewrites — index.html — and assembles the site, pointing a single <base> at
    # ``exports/<key>/`` so the assets serve from the CDN untouched: at the commit sha the
    # sync returned, when the caller pinned one (docs/publish.lock), so a later
    # re-publish can't swap assets under already-built HTML. This keeps the
    # heavy/authenticated half (export, which needs the data + a write token) on the
    # agent, and the deterministic half (link resolution, <base>) read-only in CI — no
    # cas/<sha>, no publish() copy, no accumulation on the head (names overwrite in
    # place; history holds the pinned revisions).

    def export_base(self, key: str, *, revision: str | None = None) -> str:
        """The ``<base href>`` a published report's relative ``_assets/`` resolve against.

        With *revision* (a commit sha from :meth:`sync_export`) the base is **immutable**: it serves the bundle exactly as that publish left it, so a later re-publish — from a branch, before its code merges — can't swap assets under HTML already built against it. Without one it tracks the branch head (bucket exports have no history, so there a revision is meaningless and ignored).
        """
        if self.publish_repo is not None:
            return f"https://huggingface.co/datasets/{self.publish_repo}/resolve/{revision or 'main'}/exports/{key}/"
        return f"https://huggingface.co/buckets/{self.bucket}/resolve/exports/{key}/"

    def sync_export(self, local_dir: Path, key: str) -> str | None:
        """Mirror a report's local export dir to ``exports/<key>/`` (delete stale).

        rsync-like — a re-export overwrites in place and drops assets the report no longer references — with the per-report bundle as the sync unit. On the bucket that's ``sync_bucket``; on the dataset repo it's one commit whose ``delete_patterns`` prunes the bundle's now-absent files.

        Returns the commit sha the bundle now lives at (repo mode) — the immutable revision a build can pin its ``<base>`` to — or ``None`` on a bucket, which keeps no history. An identical re-publish creates no commit (huggingface_hub drops no-op operations) and returns the current head, so publishing unchanged content never invalidates anything.
        """
        if self.publish_repo is not None:
            info = self.api.upload_folder(
                folder_path=str(local_dir),
                path_in_repo=f"exports/{key}",
                repo_id=self.publish_repo,
                repo_type="dataset",
                delete_patterns="*",
                commit_message=f"export {key}",
            )
            return info.oid
        self.api.sync_bucket(source=str(local_dir), dest=f"hf://buckets/{self.bucket}/exports/{key}", delete=True)
        return None

    def read_export_html(self, key: str, *, revision: str | None = None) -> str | None:
        """The synced bundle's ``index.html`` for *key*, or ``None`` if nothing is synced.

        The build reads exports back this way — read-only, no notebook execution — so a report missing here just means it hasn't been ``./go publish``ed yet. Only the HTML travels: the page keeps its ``_assets/`` links relative and the build points one ``<base>`` (see :meth:`export_base`) at the bundle on the CDN, so the figure bytes are already served from where they live. Pass the *revision* the build will serve so the HTML matches the assets that revision pins; ``None`` reads the head.

        Safe to call concurrently — the site build fetches its reports in one wave.
        """
        path = f"exports/{key}/index.html"
        return self._read_repo_file(path, revision) if self.publish_repo is not None else self._read_bucket_file(path)

    def _read_repo_file(self, path: str, revision: str | None) -> str | None:
        """One text file off the publish repo, or ``None`` if the repo has no such path.

        The not-found error *is* the existence answer, so catching it does the work a separate ``file_exists`` probe would — one round trip per file rather than two. Anything else (bad revision, missing repo, rejected token) is a real misconfiguration and propagates.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        repo = self.publish_repo
        assert repo is not None  # only reached from read_export_html when the publish tier is a repo
        try:
            local = hf_hub_download(
                repo_id=repo,
                filename=path,
                repo_type="dataset",
                revision=revision,
                token=self._token,
            )
        except EntryNotFoundError:
            return None
        return Path(local).read_text("utf-8")

    def _read_bucket_file(self, path: str) -> str | None:
        """One text file off the bucket, or ``None`` if it isn't there.

        Buckets keep no history, so there's no revision to read at — the head is all there is (see :meth:`export_base`).
        """
        infos = self._paths_info([path])
        if path not in infos:
            return None
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "file"
            self.api.download_bucket_files(self._cas, files=[(infos[path], str(out))])
            return out.read_text("utf-8")

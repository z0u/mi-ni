"""
Content-addressed artifact storage for experiments.

A step's *result* (the small thing a memo record holds) and its *artifacts* (the
large bytes a result points at) want different homes. Today a step that writes a
file returns a ``Path``, which pickles a *location* into the result — and that
location lives in a volume that may have evaporated by the time another process,
another experiment, or a report reads the result back.

This module fixes the asymmetry. A step ``put``s its bytes into a content-addressed
store and returns an :class:`Artifact` — a small, location-free *handle* (a sha,
a size, a logical name). The handle pickles durably into the result, and anyone
holding it can ``get`` the bytes back from the store regardless of where they run.

Two properties make this more than a tidy file copy:

- **The store is project-scoped, not experiment-scoped.** Blobs are keyed by
  content (``cas/<sha256>``), so identical bytes coincide and distinct bytes
  diverge — across experiments, for free. A small mutable *ref* layer
  (``name -> Artifact``) names views over the immutable blobs (the git
  objects-and-refs split), which is how one experiment hands an asset to another
  by a stable name (:func:`set_ref` / :func:`get_ref`).
- **Handles stabilize downstream keys.** Passing a ``Path`` into the next step
  would fingerprint it by location; passing an ``Artifact`` fingerprints it by
  content, so a consumer's memo key only moves when the bytes actually change.

Steps reach the store the way they reach the data dir — through a context var the
worker enters — so ``from mini.store import put, get`` works inside any step::

    from mini.store import put, get_ref

    def extract_features(cfg) -> Artifact:
        cache = get_data_dir() / 'acts'
        run_model(cfg, into=cache)
        return put(cache, name='activations')   # hashed into the store; handle returned

The backend is swappable behind :class:`Store`. :class:`LocalStore` (a ``cas/``
tree on disk) is the boring default and needs no network; a bucket- or repo-backed
store for web-reachable :meth:`~Store.publish` slots in behind the same handle.
"""

from __future__ import annotations

import contextvars
import gc
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import tomllib
import types
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

__all__ = [
    "Artifact",
    "BlobStat",
    "artifact_shas",
    "StaleWriteError",
    "Store",
    "LocalStore",
    "get_store",
    "store_context",
    "put",
    "get",
    "publish",
    "set_ref",
    "get_ref",
    "producer_context",
    "resolved_refs_context",
    "store_root_for",
    "store_for",
    "project_store",
    "store_bucket",
    "publish_repo",
    "STORE_BUCKET_ENV",
    "PUBLISH_REPO_ENV",
]

# Env var naming the project's Hugging Face bucket — an *override* for the
# `[tool.mini] store-bucket` committed in pyproject.toml (see `store_bucket`).
STORE_BUCKET_ENV = "MINI_STORE_BUCKET"

# Env var naming the project's Hugging Face *dataset repo* for the public,
# versioned publish tier — an override for `[tool.mini] publish-repo`. Unset →
# publish/exports stay in the (durable) bucket, as before (see `publish_repo`).
PUBLISH_REPO_ENV = "MINI_PUBLISH_REPO"

_CHUNK = 1 << 20  # 1 MiB streaming-hash chunk

log = logging.getLogger(__name__)


class StaleWriteError(RuntimeError):
    """A superseded attempt tried a mutable-name write (``set_ref`` / ``publish``).

    Raised inside a step whose attempt generation is no longer current — the task
    was relaunched or cancelled while this worker ran. CAS blobs are immune
    (write-once-by-hash), but a name write from a stale worker would silently
    last-writer-win its successor's, so the worker is stopped loudly instead. See
    the fence in :func:`mini._taskworker.execute_task`.
    """


# ---------------------------------------------------------------------------
# The handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    """A small, location-free handle to immutable bytes in a :class:`Store`.

    It carries enough to *resolve* the bytes (``sha256``) and to *serve* them
    (``name`` carries the extension; ``media_type`` overrides the guess) without
    carrying *where* they live — which is what lets it pickle durably into a
    result and fingerprint a downstream step by content rather than path.

    ``kind='tree'`` is a manifest: ``children`` are themselves artifacts (each its
    own blob), so a directory of many small files dedups per-file and resolves one
    child without pulling the set. A tree's own ``sha256`` hashes its manifest, so
    two identical directories still coincide.
    """

    sha256: str
    size: int
    name: str
    media_type: str | None = None
    kind: Literal["file", "tree"] = "file"
    children: tuple[Artifact, ...] = field(default_factory=tuple)

    @property
    def content_type(self) -> str:
        """The MIME type to serve this as — explicit ``media_type`` or guessed from ``name``."""
        if self.media_type:
            return self.media_type
        guessed, _ = mimetypes.guess_type(self.name)
        return guessed or "application/octet-stream"

    def to_dict(self) -> dict:
        """A JSON-canonical dict (recurses into ``children``) for ref storage."""
        d: dict = {"sha256": self.sha256, "size": self.size, "name": self.name, "kind": self.kind}
        if self.media_type:
            d["media_type"] = self.media_type
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Artifact:
        return cls(
            sha256=d["sha256"],
            size=d["size"],
            name=d["name"],
            media_type=d.get("media_type"),
            kind=d.get("kind", "file"),
            children=tuple(cls.from_dict(c) for c in d.get("children", ())),
        )


def _cas_key(sha256: str) -> str:
    """A blob's path within the store, sharded by a two-char prefix (``cas/ab/abcd…``).

    Git and Git-LFS both fan blobs out under a short prefix dir so the CAS never
    becomes one flat directory of thousands of entries — slow to list on disk and
    unwieldy in a bucket's web UI. One level of two hex chars (256 buckets) is
    plenty at our scale; the full sha still names the file, so it stays the id.
    """
    return f"cas/{sha256[:2]}/{sha256}"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _tree_sha(children: tuple[Artifact, ...]) -> str:
    """A stable content id for a manifest: hash the sorted ``(name, sha)`` pairs."""
    manifest = "\n".join(f"{c.name}\t{c.sha256}" for c in sorted(children, key=lambda c: c.name))
    return _hash_bytes(manifest.encode())


# Object kinds a result's data graph never hides an Artifact behind: crossing
# them would drag in whole modules (a callable's globals) for no extra recall.
_OPAQUE = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.CodeType, type)


def artifact_shas(obj: Any) -> set[str]:
    """Every *blob* sha reachable from *obj* — the reference set GC marks from.

    Walks the full object graph (``gc.get_referents``: containers, dataclass
    fields, instance dicts — anything a result value can nest handles in),
    pruned at code/module/class boundaries. A tree's own sha names a manifest,
    not a stored blob, so only ``file`` artifacts (including tree children)
    contribute.
    """
    shas: set[str] = set()
    seen: set[int] = set()
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, (str, bytes, int, float, bool, type(None))) or id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, Artifact):
            if o.kind == "file":
                shas.add(o.sha256)
            stack.extend(o.children)
        elif not isinstance(o, _OPAQUE):
            stack.extend(gc.get_referents(o))
    return shas


# ---------------------------------------------------------------------------
# Ref provenance: who wrote a name, and who read it
#
# The store is project-shared and knows nothing about experiments, so producer
# identity arrives ambiently: the task worker wraps a step in `producer_context`
# and `set_ref` stamps that identity into the ref payload. The consumer side is
# the mirror image — `resolved_refs_context` collects every ref a step resolves
# (with its stamped producer), which is how a run's upstream experiments are
# detected without declaring `Experiment(deps=[...])` by hand, and how a report
# knows which runs its data came from.
# ---------------------------------------------------------------------------

_producer: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("mini_ref_producer", default=None)
_resolved: contextvars.ContextVar[dict[str, dict[str, Any] | None] | None] = contextvars.ContextVar(
    "mini_resolved_refs", default=None
)


@contextmanager
def producer_context(info: dict[str, Any]) -> Iterator[None]:
    """Bind *info* as the identity ``set_ref`` stamps into refs written in this context.

    Set by the task worker around a step (experiment name, task key, and the run's
    code state from its stored lineage). Refs written outside any producer context
    are simply unstamped — capture degrades, it never blocks a write.
    """
    token = _producer.set(info)
    try:
        yield
    finally:
        _producer.reset(token)


@contextmanager
def resolved_refs_context(seen: dict[str, dict[str, Any] | None]) -> Iterator[None]:
    """Collect every ref resolved in this context into *seen* (name → stamped producer).

    The task worker reads the collected set back after the step to record which
    upstream experiments actually fed it (``upstream_refs`` on the task record).
    """
    token = _resolved.set(seen)
    try:
        yield
    finally:
        _resolved.reset(token)


def _note_resolution(name: str, producer: dict[str, Any] | None) -> None:
    """Fan a ref resolution out to whoever is listening — never let it break a read."""
    if (seen := _resolved.get()) is not None:
        seen[name] = producer
    try:  # a rendering report tracks resolutions through its active publisher
        from mini.reports import current_publisher

        if pub := current_publisher():
            pub.note_ref(name, producer)
    except Exception:
        log.debug("failed to note ref resolution for %r", name, exc_info=True)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlobStat:
    """One stored blob, as the GC sweep sees it (see ``Store.list_blobs``)."""

    sha256: str
    size: int
    modified_at: float | None  # epoch seconds; None = unknown, treated as too new to sweep


class Store(ABC):
    """A content-addressed blob store with a small mutable ref layer.

    Backends implement the four blob/ref primitives below; the high-level
    :meth:`put` / :meth:`get` (including tree fan-out) and JSON ref handling are
    shared. ``put`` is idempotent — hash first, skip the write if :meth:`has` —
    so re-runs and cross-step duplicates cost nothing.
    """

    # -- backend primitives ---------------------------------------------------

    @abstractmethod
    def has(self, sha256: str) -> bool:
        """Whether a blob with this content hash is already stored."""

    @abstractmethod
    def _write_blob(self, sha256: str, src: Path) -> None:
        """Idempotently store the file at *src* under *sha256* (treat blobs as immutable)."""

    @abstractmethod
    def _read_blob(self, sha256: str, dest: Path) -> None:
        """Materialize the blob *sha256* to the file *dest* (parent dirs exist)."""

    @abstractmethod
    def _write_ref(self, name: str, payload: str) -> None:
        """Set the mutable ref *name* to *payload* (last writer wins)."""

    @abstractmethod
    def _read_ref(self, name: str) -> str | None:
        """Read the ref *name*, or ``None`` if unset."""

    @abstractmethod
    def publish(self, art: Artifact, path: str) -> str:
        """Expose *art* at a named, extensioned *path* and return its URL.

        A by-hash copy to a path whose extension drives the served ``Content-Type``
        (a bare ``cas/<sha>`` has no extension, so a browser won't render it). Kept
        separate from :meth:`put` and deliberately the only outward-facing verb, so
        persisting a result never publishes it as a side effect.
        """

    # -- high-level surface (shared) ------------------------------------------

    def put(self, data: bytes | Path, *, name: str) -> Artifact:
        """Store *data* (bytes, a file, or a directory) and return its handle.

        A directory becomes a ``tree`` artifact: each file is stored as its own
        blob and the returned handle carries the manifest. ``name`` is the logical
        name (carry the extension — it sets the served media type).
        """
        if isinstance(data, (bytes, bytearray)):
            return self._put_bytes(bytes(data), name=name)
        src = Path(data)
        if src.is_dir():
            return self._put_tree(src, name=name)
        return self._put_file(src, name=name)

    def _put_bytes(self, data: bytes, *, name: str) -> Artifact:
        sha = _hash_bytes(data)
        if not self.has(sha):
            with _spill(data) as tmp:
                self._write_blob(sha, tmp)
        return Artifact(sha256=sha, size=len(data), name=name)

    def _put_file(self, src: Path, *, name: str) -> Artifact:
        sha, size = _hash_file(src)
        if not self.has(sha):
            self._write_blob(sha, src)
        return Artifact(sha256=sha, size=size, name=name)

    def _put_tree(self, src: Path, *, name: str) -> Artifact:
        children = tuple(
            self._put_file(p, name=str(p.relative_to(src).as_posix())) for p in sorted(src.rglob("*")) if p.is_file()
        )
        sha = _tree_sha(children)
        size = sum(c.size for c in children)
        return Artifact(sha256=sha, size=size, name=name, kind="tree", children=children)

    def get(self, art: Artifact, dest: Path) -> Path:
        """Materialize *art* at *dest* and return it.

        For a ``file`` artifact *dest* is the destination file; for a ``tree`` it's
        the destination directory, and the children resolve concurrently (the
        per-op latency of a remote backend overlaps rather than serializes).
        """
        dest = Path(dest)
        if art.kind == "tree":
            dest.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=min(8, len(art.children) or 1)) as ex:
                list(ex.map(lambda c: self.get(c, dest / c.name), art.children))
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._read_blob(art.sha256, dest)
        return dest

    def set_ref(self, name: str, art: Artifact) -> None:
        """Point the mutable name *name* at *art* — the cross-experiment by-name handle.

        When an ambient :func:`producer_context` is set (the task worker binds one),
        the writer's identity rides the payload as a ``producer`` key —
        ``Artifact.from_dict`` ignores it, so old readers are unaffected.
        """
        payload = art.to_dict()
        if producer := _producer.get():
            payload["producer"] = {**producer, "written_at": datetime.now(timezone.utc).isoformat()}
        self._write_ref(name, json.dumps(payload, sort_keys=True))

    def get_ref(self, name: str) -> Artifact | None:
        """Resolve the name *name* to its artifact handle, or ``None`` if unset.

        Each resolution is announced (:func:`_note_resolution`) with the producer
        stamped at ``set_ref`` time, so a consuming run records its upstream
        experiments and a rendering report can cite where its data came from.
        """
        payload = self._read_ref(name)
        if payload is None:
            return None
        d = json.loads(payload)
        _note_resolution(name, d.get("producer"))
        return Artifact.from_dict(d)

    def ref_producer(self, name: str) -> dict[str, Any] | None:
        """The producer stamped on ref *name* (see :meth:`set_ref`), or ``None``."""
        payload = self._read_ref(name)
        return json.loads(payload).get("producer") if payload is not None else None

    # -- gc surface (optional capability) --------------------------------------
    #
    # Not abstract: only the durable backends (LocalStore, HFStore) can sweep;
    # wrappers and read-only views (_FencedStore, ModalVolumeStore) need not
    # pretend to. ``mini gc --store`` checks for NotImplementedError and says so.

    def list_blobs(self) -> Iterator[BlobStat]:
        """Every blob in the CAS, with size and last-modified (the sweep candidates)."""
        raise NotImplementedError(f"{type(self).__name__} does not support gc")

    def delete_blobs(self, sha256s: Iterable[str]) -> None:
        """Remove blobs from the CAS (and any warm cache, so ``has`` cannot lie)."""
        raise NotImplementedError(f"{type(self).__name__} does not support gc")

    def list_refs(self) -> list[str]:
        """Every ref name currently set (each is a GC root)."""
        raise NotImplementedError(f"{type(self).__name__} does not support gc")


@contextmanager
def _spill(data: bytes) -> Iterator[Path]:
    """Write *data* to a short-lived temp file (so a bytes ``put`` reuses the file path)."""
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data)
        tmp = Path(f.name)
    try:
        yield tmp
    finally:
        tmp.unlink(missing_ok=True)


def store_root_for(data_dir: Path | str) -> Path:
    """The project-scoped store root that sits beside an experiment's *data_dir*.

    A volume path is ``<data_root>/<experiment>``, so its parent is the project
    root and ``<parent>/store`` is shared by every experiment — content-addressed,
    so identical bytes coincide and a named ref handed off by one experiment
    resolves in another. Derived from the path (not the cwd), so a detached worker
    under its own cwd lands on the same store.
    """
    return Path(data_dir).parent / "store"


def _project_config() -> dict:
    """``[tool.mini]`` from the nearest ``pyproject.toml`` walking up from cwd, or ``{}``.

    Read lazily off the live cwd (like :func:`~mini.runs.data_root`) so a ``chdir``
    — or a test in a tmp dir — resolves the right project, and nothing parses TOML
    at import time.
    """
    cwd = Path.cwd().resolve()
    for d in (cwd, *cwd.parents):
        pp = d / "pyproject.toml"
        if pp.exists():
            try:
                return tomllib.loads(pp.read_text()).get("tool", {}).get("mini", {})
            except OSError, tomllib.TOMLDecodeError:
                return {}
    return {}


def store_bucket() -> str | None:
    """The configured Hugging Face bucket (``namespace/name``), or ``None`` for local.

    Resolution order: the ``MINI_STORE_BUCKET`` env var first (so CI or a one-off
    shell can override), else ``[tool.mini] store-bucket`` in ``pyproject.toml`` —
    so the project's default *travels with the repo*, set once and shared by every
    checkout, Modal worker, and CI run rather than re-set in three places. The
    bucket name isn't a secret; the token still lives in the env / ``hf`` cache.
    """
    return os.environ.get(STORE_BUCKET_ENV) or _project_config().get("store-bucket")


def publish_repo() -> str | None:
    """The configured Hugging Face *dataset repo* for the publish tier, or ``None``.

    When set, :meth:`~mini.hf_store.HFStore.publish` and the report-export methods
    target this public, git-backed dataset repo instead of the durable bucket — so
    the CAS bucket can be private (persisting an artifact never makes its bytes
    world-readable) and published views get real history (a citation pins to a
    commit sha). Unset → publish/exports stay in the bucket, the single-store
    default. Resolution mirrors :func:`store_bucket` (``MINI_PUBLISH_REPO`` env
    first, else ``[tool.mini] publish-repo``); the repo id isn't a secret.
    """
    return os.environ.get(PUBLISH_REPO_ENV) or _project_config().get("publish-repo")


def _hf_token() -> str | None:
    """The Hugging Face token from the env or the ``hf auth login`` cache, or ``None``."""
    if tok := os.environ.get("HF_TOKEN"):
        return tok
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def store_for(root: Path | str, *, cache_root: Path | str | None = None) -> Store:
    """The project store for a given local *root* — bucket-backed if configured.

    When a bucket is configured (:func:`store_bucket`) *and* a Hugging Face token is
    available, the durable store is the shared bucket, warm-cached locally at
    *cache_root* (default: ``store-cache/hf`` beside *root*); otherwise it's a
    :class:`LocalStore` rooted at *root*. One switch flips every put/get/publish —
    in a step, a report, or a worker — from on-disk to shared-and-web-reachable.

    Pass *cache_root* when *root*'s neighbourhood is the wrong home for cached
    bytes: a Modal worker points it at container-local disk, so the cache isn't
    committed to the Volume alongside results — the bucket already holds the
    durable copy, and a committed shadow would store every artifact twice.

    A configured bucket with *no* token (someone trying the repo in Codespaces, or
    a fresh checkout before ``./go auth``) falls back to the local store with a
    warning rather than failing mid-run — the bucket name travels with the repo,
    but using it needs auth the trier doesn't have yet.

    A :func:`publish_repo` alone (no bucket) also yields an :class:`HFStore`, but a
    CAS-less one: it serves ``publish``/exports from the dataset repo and errors on
    put/get/ref. That's the read-only site build's store — it reads exports off the
    public repo, so CI needs only ``MINI_PUBLISH_REPO``, not the private bucket (#38).
    """
    root = Path(root)
    bucket, repo = store_bucket(), publish_repo()
    token = _hf_token()
    # A configured bucket needs a token (the CAS is usually private), so fall back to
    # the local store rather than failing a fresh checkout mid-run. A publish-repo on
    # its own needs no bucket — the read-only site build serves exports straight from
    # the (public) dataset repo — so it's enough by itself to build the networked
    # store, and CI can point at the repo without also naming the private bucket (#38).
    if bucket and not token:
        log.warning(
            "store-bucket %r is configured but no Hugging Face token was found — using the local "
            "store instead. Run `./go auth` (or set HF_TOKEN) to read/write the shared bucket.",
            bucket,
        )
        return LocalStore(root)
    if bucket or repo:
        from mini.hf_store import HFStore

        cache = LocalStore(cache_root if cache_root is not None else root.parent / "store-cache" / "hf")
        return HFStore(bucket, cache=cache, token=token, publish_repo=repo)
    return LocalStore(root)


def project_store() -> Store:
    """The project-scoped artifact :class:`Store`, resolved from the project root.

    The artifact store is one-per-project (a ``store/`` beside the experiment
    volumes under ``.mini``), so it needs no experiment name. Use this *outside* a
    step — a report or notebook reading a shared ref, a driver publishing one —
    where there's no ambient store; *inside* a step the worker already binds the
    same store, so bare :func:`put` / :func:`get` / :func:`get_ref` resolve here.
    """
    from mini.runs import data_root

    return store_for(data_root() / "store")


class LocalStore(Store):
    """A ``cas/<ab>/<sha256>`` blob tree on local disk, with file-backed refs and views.

    The boring default: no network, immutability enforced by write-once-by-hash.
    ``publish`` copies a blob to ``published/<path>`` and returns a ``file://`` URL
    — the same shape a bucket-backed store returns as an ``https://`` resolve URL,
    so a report reads one ``url`` either way.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.refs = self.root / "refs"
        self.published = self.root / "published"

    def _blob_path(self, sha256: str) -> Path:
        return self.root / _cas_key(sha256)

    def has(self, sha256: str) -> bool:
        return self._blob_path(sha256).exists()

    def _write_blob(self, sha256: str, src: Path) -> None:
        dest = self._blob_path(sha256)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():  # immutable: another writer won the race; bytes are identical by hash
            return
        tmp = dest.with_name(f"{sha256}.tmp.{src.stat().st_ino}")
        shutil.copyfile(src, tmp)  # copy (never hardlink): a caller mutating dest must not corrupt the CAS
        tmp.replace(dest)  # atomic publish into the CAS

    def _read_blob(self, sha256: str, dest: Path) -> None:
        shutil.copyfile(self._blob_path(sha256), dest)

    def _ref_path(self, name: str) -> Path:
        return self.refs / f"{name}.json"

    def _write_ref(self, name: str, payload: str) -> None:
        p = self._ref_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(payload)
        tmp.replace(p)

    def _read_ref(self, name: str) -> str | None:
        p = self._ref_path(name)
        return p.read_text() if p.exists() else None

    def publish(self, art: Artifact, path: str) -> str:
        if art.kind == "tree":
            raise ValueError("publish a single file (resolve a tree first, or publish its children)")
        dest = self.published / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._read_blob(art.sha256, dest)
        return dest.resolve().as_uri()

    # -- gc --------------------------------------------------------------------

    def list_blobs(self) -> Iterator[BlobStat]:
        cas = self.root / "cas"
        if not cas.is_dir():
            return
        for p in sorted(cas.glob("*/*")):
            # Only true blob names count: a .tmp from a crashed _write_blob is
            # not part of the CAS (and never will be — the rename lost).
            if p.is_file() and len(p.name) == 64 and set(p.name) <= set("0123456789abcdef"):
                st = p.stat()
                yield BlobStat(p.name, st.st_size, st.st_mtime)

    def delete_blobs(self, sha256s: Iterable[str]) -> None:
        for sha in sha256s:
            self._blob_path(sha).unlink(missing_ok=True)

    def list_refs(self) -> list[str]:
        if not self.refs.is_dir():
            return []
        return sorted(p.relative_to(self.refs).as_posix()[: -len(".json")] for p in self.refs.rglob("*.json"))


# ---------------------------------------------------------------------------
# Ambient store (the get_data_dir pattern)
# ---------------------------------------------------------------------------

_store: contextvars.ContextVar[Store | None] = contextvars.ContextVar("mini_store", default=None)


@contextmanager
def store_context(store: Store) -> Iterator[None]:
    """Bind *store* as the ambient store for :func:`put` / :func:`get` in this context."""
    token = _store.set(store)
    try:
        yield
    finally:
        _store.reset(token)


def get_store() -> Store:
    """The ambient :class:`Store`, set by the apparatus around a step (or a report).

    Raises if called outside a store context — the same contract as
    :func:`~mini.volume.get_data_dir`.
    """
    s = _store.get()
    if s is None:
        raise RuntimeError(
            "No store configured. put()/get() must run inside a step launched by an "
            "Apparatus, or under an explicit store_context(...)."
        )
    return s


def put(data: bytes | Path, *, name: str) -> Artifact:
    """Store *data* in the ambient store and return its handle. See :meth:`Store.put`."""
    return get_store().put(data, name=name)


def get(art: Artifact, dest: Path) -> Path:
    """Materialize *art* from the ambient store at *dest*. See :meth:`Store.get`."""
    return get_store().get(art, dest)


def publish(art: Artifact, path: str) -> str:
    """Publish *art* at a named *path* via the ambient store. See :meth:`Store.publish`."""
    return get_store().publish(art, path)


def set_ref(name: str, art: Artifact) -> None:
    """Point a name at *art* in the ambient store — the cross-experiment handle."""
    get_store().set_ref(name, art)


def get_ref(name: str) -> Artifact | None:
    """Resolve a name to its artifact in the ambient store, or ``None``."""
    return get_store().get_ref(name)

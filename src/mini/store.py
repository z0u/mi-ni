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
import hashlib
import json
import mimetypes
import os
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

__all__ = [
    'Artifact',
    'Store',
    'LocalStore',
    'get_store',
    'store_context',
    'put',
    'get',
    'publish',
    'set_ref',
    'get_ref',
    'store_root_for',
    'default_store',
    'STORE_BUCKET_ENV',
]

# Env var naming the project's Hugging Face bucket. Set it (alongside HF_TOKEN)
# to make the durable/publish tier the shared bucket; unset, the store is local.
STORE_BUCKET_ENV = 'MINI_STORE_BUCKET'

_CHUNK = 1 << 20  # 1 MiB streaming-hash chunk


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
    kind: Literal['file', 'tree'] = 'file'
    children: tuple[Artifact, ...] = field(default_factory=tuple)

    @property
    def content_type(self) -> str:
        """The MIME type to serve this as — explicit ``media_type`` or guessed from ``name``."""
        if self.media_type:
            return self.media_type
        guessed, _ = mimetypes.guess_type(self.name)
        return guessed or 'application/octet-stream'

    def to_dict(self) -> dict:
        """A JSON-canonical dict (recurses into ``children``) for ref storage."""
        d: dict = {'sha256': self.sha256, 'size': self.size, 'name': self.name, 'kind': self.kind}
        if self.media_type:
            d['media_type'] = self.media_type
        if self.children:
            d['children'] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Artifact:
        return cls(
            sha256=d['sha256'],
            size=d['size'],
            name=d['name'],
            media_type=d.get('media_type'),
            kind=d.get('kind', 'file'),
            children=tuple(cls.from_dict(c) for c in d.get('children', ())),
        )


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _tree_sha(children: tuple[Artifact, ...]) -> str:
    """A stable content id for a manifest: hash the sorted ``(name, sha)`` pairs."""
    manifest = '\n'.join(f'{c.name}\t{c.sha256}' for c in sorted(children, key=lambda c: c.name))
    return _hash_bytes(manifest.encode())


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


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
            self._put_file(p, name=str(p.relative_to(src).as_posix()))
            for p in sorted(src.rglob('*'))
            if p.is_file()
        )
        sha = _tree_sha(children)
        size = sum(c.size for c in children)
        return Artifact(sha256=sha, size=size, name=name, kind='tree', children=children)

    def get(self, art: Artifact, dest: Path) -> Path:
        """Materialize *art* at *dest* and return it.

        For a ``file`` artifact *dest* is the destination file; for a ``tree`` it's
        the destination directory, and the children resolve concurrently (the
        per-op latency of a remote backend overlaps rather than serializes).
        """
        dest = Path(dest)
        if art.kind == 'tree':
            dest.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=min(8, len(art.children) or 1)) as ex:
                list(ex.map(lambda c: self.get(c, dest / c.name), art.children))
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._read_blob(art.sha256, dest)
        return dest

    def set_ref(self, name: str, art: Artifact) -> None:
        """Point the mutable name *name* at *art* — the cross-experiment by-name handle."""
        self._write_ref(name, json.dumps(art.to_dict(), sort_keys=True))

    def get_ref(self, name: str) -> Artifact | None:
        """Resolve the name *name* to its artifact handle, or ``None`` if unset."""
        payload = self._read_ref(name)
        return Artifact.from_dict(json.loads(payload)) if payload is not None else None


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
    return Path(data_dir).parent / 'store'


def default_store(root: Path | str) -> Store:
    """The project store for a given local *root* — bucket-backed if configured.

    When ``MINI_STORE_BUCKET`` is set the durable store is the shared Hugging Face
    bucket (with *root* demoted to a local warm cache); otherwise it's a
    :class:`LocalStore` rooted at *root*. One switch flips every put/get/publish —
    in a step, a report, or a worker — from on-disk to shared-and-web-reachable.
    """
    root = Path(root)
    bucket = os.environ.get(STORE_BUCKET_ENV)
    if bucket:
        from mini.hf_store import HFStore

        return HFStore(bucket, cache=LocalStore(root.parent / 'store-cache' / 'hf'))
    return LocalStore(root)


class LocalStore(Store):
    """A ``cas/<sha256>`` blob tree on local disk, with file-backed refs and views.

    The boring default: no network, immutability enforced by write-once-by-hash.
    ``publish`` copies a blob to ``published/<path>`` and returns a ``file://`` URL
    — the same shape a bucket-backed store returns as an ``https://`` resolve URL,
    so a report reads one ``url`` either way.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.cas = self.root / 'cas'
        self.refs = self.root / 'refs'
        self.published = self.root / 'published'

    def _blob_path(self, sha256: str) -> Path:
        return self.cas / sha256

    def has(self, sha256: str) -> bool:
        return self._blob_path(sha256).exists()

    def _write_blob(self, sha256: str, src: Path) -> None:
        self.cas.mkdir(parents=True, exist_ok=True)
        dest = self._blob_path(sha256)
        if dest.exists():  # immutable: another writer won the race; bytes are identical by hash
            return
        tmp = dest.with_name(f'{sha256}.tmp.{src.stat().st_ino}')
        shutil.copyfile(src, tmp)  # copy (never hardlink): a caller mutating dest must not corrupt the CAS
        tmp.replace(dest)  # atomic publish into the CAS

    def _read_blob(self, sha256: str, dest: Path) -> None:
        shutil.copyfile(self._blob_path(sha256), dest)

    def _ref_path(self, name: str) -> Path:
        return self.refs / f'{name}.json'

    def _write_ref(self, name: str, payload: str) -> None:
        p = self._ref_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.json.tmp')
        tmp.write_text(payload)
        tmp.replace(p)

    def _read_ref(self, name: str) -> str | None:
        p = self._ref_path(name)
        return p.read_text() if p.exists() else None

    def publish(self, art: Artifact, path: str) -> str:
        if art.kind == 'tree':
            raise ValueError('publish a single file (resolve a tree first, or publish its children)')
        dest = self.published / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._read_blob(art.sha256, dest)
        return dest.resolve().as_uri()


# ---------------------------------------------------------------------------
# Ambient store (the get_data_dir pattern)
# ---------------------------------------------------------------------------

_store: contextvars.ContextVar[Store | None] = contextvars.ContextVar('mini_store', default=None)


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
            'No store configured. put()/get() must run inside a step launched by an '
            'Apparatus, or under an explicit store_context(...).'
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

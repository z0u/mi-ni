"""The publish-tier split (#38): routing, not transport.

``HFStore`` keeps the CAS + refs in its bucket but sends ``publish`` and report
exports to a separate, versioned dataset repo when one is configured. These tests
inject a fake ``HfApi`` to assert *where* each verb lands and *what* URL it returns,
without touching the network — the live round trips stay in ``test_hf_store.py``
(bucket) and its ``MINI_PUBLISH_REPO``-gated repo cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mini.hf_store import HFStore
from mini.store import Artifact, LocalStore, _hash_bytes


class _Info:
    def __init__(self, xet_hash: str):
        self.xet_hash = xet_hash


class FakeApi:
    """Records calls; ``present`` toggles whether the CAS claims to hold the blob."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.present = True

    def get_bucket_paths_info(self, bucket, paths):
        self.calls.append(("get_bucket_paths_info", bucket, tuple(paths)))
        return [_Info("xhash")] if self.present else []

    def batch_bucket_files(self, bucket, **kw):
        self.calls.append(("batch_bucket_files", bucket, kw))

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))

    def file_exists(self, **kw):
        self.calls.append(("file_exists", kw))
        return True


def _store(tmp_path: Path, *, publish_repo: str | None = None) -> HFStore:
    store = HFStore("ns/bkt", cache=LocalStore(tmp_path / "cache"), token="tok", publish_repo=publish_repo)
    store._api = FakeApi()
    return store


def _cache_blob(store: HFStore, data: bytes) -> Artifact:
    """Seed the warm cache so ``has()`` and ``_local_blob()`` resolve offline."""
    sha = _hash_bytes(data)
    blob = store._cache._blob_path(sha)
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    return Artifact(sha256=sha, size=len(data), name="fig.png")


def _verbs(store: HFStore) -> list[str]:
    return [c[0] for c in store._api.calls]


def test_publish_without_repo_copies_in_bucket(tmp_path: Path):
    store = _store(tmp_path)
    art = _cache_blob(store, b"\x89PNG")
    url = store.publish(art, "_x/fig.png")
    assert url == "https://huggingface.co/buckets/ns/bkt/resolve/published/_x/fig.png"
    assert "batch_bucket_files" in _verbs(store)  # server-side by-hash copy, in the bucket
    assert "upload_file" not in _verbs(store)


def test_publish_with_repo_uploads_to_dataset(tmp_path: Path):
    store = _store(tmp_path, publish_repo="ns/pub")
    art = _cache_blob(store, b"\x89PNG")  # in the CAS, so has() is satisfied from the warm cache
    url = store.publish(art, "_x/fig.png")
    assert url == "https://huggingface.co/datasets/ns/pub/resolve/main/published/_x/fig.png"
    uploads = [c[1] for c in store._api.calls if c[0] == "upload_file"]
    assert len(uploads) == 1
    kw = uploads[0]
    assert kw["repo_id"] == "ns/pub"
    assert kw["repo_type"] == "dataset"
    assert kw["path_in_repo"] == "published/_x/fig.png"
    assert "batch_bucket_files" not in _verbs(store)  # nothing lands in the CAS bucket


def test_publish_with_repo_needs_the_blob_in_the_cas(tmp_path: Path):
    store = _store(tmp_path, publish_repo="ns/pub")
    store._api.present = False  # neither cache nor bucket holds it
    art = Artifact(sha256="0" * 64, size=1, name="fig.png")
    with pytest.raises(FileNotFoundError):
        store.publish(art, "_x/fig.png")


def test_export_routes_to_dataset_when_repo_set(tmp_path: Path):
    store = _store(tmp_path, publish_repo="ns/pub")
    assert store.export_base("k") == "https://huggingface.co/datasets/ns/pub/resolve/main/exports/k/"
    src = tmp_path / "exp"
    src.mkdir()
    (src / "index.html").write_text("x")
    store.sync_export(src, "k")
    folders = [c[1] for c in store._api.calls if c[0] == "upload_folder"]
    assert len(folders) == 1
    assert folders[0]["path_in_repo"] == "exports/k"
    assert folders[0]["delete_patterns"] == "*"  # rsync-like: prune assets the report dropped


def test_export_base_uses_bucket_without_repo(tmp_path: Path):
    store = _store(tmp_path)
    assert store.export_base("k") == "https://huggingface.co/buckets/ns/bkt/resolve/exports/k/"

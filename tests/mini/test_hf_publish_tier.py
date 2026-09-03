"""The publish-tier split (#38): routing, not transport.

``HFStore`` keeps the CAS + refs in its bucket but sends ``publish`` and report exports to a separate, versioned dataset repo when one is configured. These tests inject a fake ``HfApi`` to assert *where* each verb lands and *what* URL it returns, without touching the network — the live round trips stay in ``test_hf_store.py`` (bucket) and its ``MINI_PUBLISH_REPO``-gated repo cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mini.hf_store import HFStore
from mini.store import Artifact, LocalStore, _hash_bytes


class _Info:
    def __init__(self, path: str = "", xet_hash: str = "xhash"):
        self.path = path
        self.xet_hash = xet_hash


class _Commit:
    def __init__(self, oid: str):
        self.oid = oid


FAKE_OID = "c0ffee" * 6 + "beef"  # 40 hex chars, like a real commit sha


class FakeApi:
    """Records calls; ``present`` toggles whether the bucket claims to hold the path."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.present = True
        self.contents: dict[str, bytes] = {}  # bucket path → bytes, for download_bucket_files

    def get_bucket_paths_info(self, bucket, paths):
        self.calls.append(("get_bucket_paths_info", bucket, tuple(paths)))
        return [_Info(p) for p in paths] if self.present else []

    def download_bucket_files(self, bucket, files):
        self.calls.append(("download_bucket_files", bucket, tuple(info.path for info, _ in files)))
        for info, dest in files:
            Path(dest).write_bytes(self.contents.get(info.path, b""))

    def batch_bucket_files(self, bucket, **kw):
        self.calls.append(("batch_bucket_files", bucket, kw))

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw))
        return _Commit(FAKE_OID)

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))
        return _Commit(FAKE_OID)


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
    # Pinned to the commit the upload made, not the branch — the URL can't be swapped later.
    assert url == f"https://huggingface.co/datasets/ns/pub/resolve/{FAKE_OID}/published/_x/fig.png"
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
    assert store.sync_export(src, "k") == FAKE_OID  # the revision the caller pins
    folders = [c[1] for c in store._api.calls if c[0] == "upload_folder"]
    assert len(folders) == 1
    assert folders[0]["path_in_repo"] == "exports/k"
    assert folders[0]["delete_patterns"] == "*"  # rsync-like: prune assets the report dropped
    assert store.export_base("k", revision=FAKE_OID) == (
        f"https://huggingface.co/datasets/ns/pub/resolve/{FAKE_OID}/exports/k/"
    )


def _fake_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str | None) -> dict:
    """Stand in for ``hf_hub_download``: serve *body*, or 404 when it's ``None``."""
    from huggingface_hub.errors import EntryNotFoundError

    seen: dict = {}

    def fake(**kw):
        seen.update(kw)
        if body is None:
            raise EntryNotFoundError("404")
        out = tmp_path / "dl" / Path(kw["filename"]).name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body)
        return str(out)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
    return seen


def test_read_export_html_pulls_one_file_at_the_pinned_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The build rewrites index.html and leaves the assets on the CDN behind a <base>, so
    # the read asks for that one file — pulling the bundle would fetch figures to discard.
    store = _store(tmp_path, publish_repo="ns/pub")
    seen = _fake_download(tmp_path, monkeypatch, "pinned")
    assert store.read_export_html("k", revision=FAKE_OID) == "pinned"
    assert seen["filename"] == "exports/k/index.html"
    assert seen["revision"] == FAKE_OID  # read at the same revision the <base> will serve
    assert seen["repo_id"] == "ns/pub"


def test_read_export_html_is_none_when_nothing_is_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Not-found is the answer to "is it published?", so it costs no extra round trip.
    store = _store(tmp_path, publish_repo="ns/pub")
    _fake_download(tmp_path, monkeypatch, None)
    assert store.read_export_html("k") is None
    assert store._api.calls == []  # no separate existence probe against the repo


def test_export_serves_from_the_bucket_without_a_repo(tmp_path: Path):
    store = _store(tmp_path)  # no publish repo: exports live in the bucket
    assert store.export_base("k") == "https://huggingface.co/buckets/ns/bkt/resolve/exports/k/"
    # Buckets keep no history — a revision is meaningless there and ignored.
    assert store.export_base("k", revision=FAKE_OID) == "https://huggingface.co/buckets/ns/bkt/resolve/exports/k/"

    store._api.contents["exports/k/index.html"] = b"<html>bucketed</html>"
    assert store.read_export_html("k") == "<html>bucketed</html>"
    pulled = [c for c in store._api.calls if c[0] == "download_bucket_files"]
    assert pulled == [("download_bucket_files", "ns/bkt", ("exports/k/index.html",))]  # just the HTML

    store._api.present = False  # nothing synced under the key
    assert store.read_export_html("k") is None

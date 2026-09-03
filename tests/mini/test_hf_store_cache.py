"""Offline unit tests for the HFStore warm cache — no network, fake ``api``.

These exercise the local cache path (``_local_blob`` / ``_pull_blobs``) and the batched read surface (``get_refs`` / ``get_many``) with a stub bucket API, so they run without a bucket or token (unlike the integration suite in ``test_hf_store.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from mini.hf_store import HFStore
from mini.store import Artifact, LocalStore, _cas_key


@dataclass(frozen=True)
class FakeInfo:
    """Stands in for ``BucketFile``: the only field the store reads back is ``path``."""

    path: str


class FakeApi:
    """Stands in for ``HfApi``: serves blob bytes from a dict, or fails a download."""

    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs  # bucket path -> bytes
        self.fail_next = False
        self.calls = 0  # download_bucket_files invocations (round trips, roughly)
        self.downloads = 0  # individual files served

    def get_bucket_paths_info(self, cas, paths):  # noqa: ANN001 — mirrors HfApi's signature
        return [FakeInfo(p) for p in paths if p in self.blobs]

    def download_bucket_files(self, cas, files):  # noqa: ANN001 — mirrors HfApi's signature
        self.calls += 1
        for info, dest in files:
            self.downloads += 1
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if self.fail_next:
                # Simulate a Ctrl-C mid-download: the tool has created the file but
                # written nothing before the interrupt lands.
                dest.write_bytes(b"")
                self.fail_next = False
                raise KeyboardInterrupt("interrupted download")
            dest.write_bytes(self.blobs[info.path])


def _store(tmp_path: Path, blobs: dict[str, bytes]) -> tuple[HFStore, FakeApi]:
    store = HFStore("ns/bucket", cache=LocalStore(tmp_path / "cache"))
    api = FakeApi(blobs)
    store._api = api  # inject the stub, bypassing the real HfApi
    return store, api


def test_interrupted_download_leaves_no_poisoned_cache_entry(tmp_path: Path):
    payload = b"the real checkpoint bytes"
    sha = "a" * 64
    blobs = {_cas_key(sha): payload}
    store, api = _store(tmp_path, blobs)

    # First pull is interrupted after creating an empty file.
    api.fail_next = True
    with pytest.raises(KeyboardInterrupt):
        store._local_blob(sha)

    # The regression guard: no 0-byte file is left at the final cache path, so the
    # cache doesn't "hit" on a corrupt entry — and no stray temp file lingers either.
    assert not store._cache._blob_path(sha).exists()
    assert list((tmp_path / "cache").rglob("*.tmp.*")) == []

    # A retry re-pulls and serves the real bytes.
    out = store._local_blob(sha)
    assert out.read_bytes() == payload


def test_second_read_hits_the_warm_cache(tmp_path: Path):
    payload = b"cached once, served twice"
    sha = "b" * 64
    store, api = _store(tmp_path, {_cas_key(sha): payload})

    assert store._local_blob(sha).read_bytes() == payload
    assert store._local_blob(sha).read_bytes() == payload
    assert api.downloads == 1  # the second read never touches the bucket


def test_missing_blob_raises_rather_than_downloading_the_rest(tmp_path: Path):
    """A sha absent from the bucket is an integrity failure, not a silent skip."""
    present = "c" * 64
    absent = "d" * 64
    store, api = _store(tmp_path, {_cas_key(present): b"here"})

    with pytest.raises(FileNotFoundError, match=absent[:12]):
        store._pull_blobs([present, absent])
    assert api.calls == 0  # detected at paths-info time, before any transfer


def test_tree_get_pulls_all_children_in_one_call(tmp_path: Path):
    """The per-call round-trip floor is paid once per tree, not once per child."""
    kids = {f"{c}" * 64: f"shard {c}".encode() for c in "abc"}
    store, api = _store(tmp_path, {_cas_key(s): b for s, b in kids.items()})
    children = tuple(Artifact(sha256=s, size=len(b), name=f"{b[-1:].decode()}.npy") for s, b in kids.items())
    tree = Artifact(sha256="e" * 64, size=3, name="ckpt", kind="tree", children=children)

    out = store.get(tree, tmp_path / "out")
    assert api.calls == 1
    assert sorted(p.name for p in out.iterdir()) == ["a.npy", "b.npy", "c.npy"]
    assert (out / "a.npy").read_bytes() == b"shard a"


def test_get_many_batches_files_and_trees_together(tmp_path: Path):
    kids = {f"{c}" * 64: f"blob {c}".encode() for c in "abcd"}
    store, api = _store(tmp_path, {_cas_key(s): b for s, b in kids.items()})
    shas = list(kids)
    tree = Artifact(
        sha256="f" * 64,
        size=2,
        name="tree",
        kind="tree",
        children=(
            Artifact(sha256=shas[0], size=6, name="x.bin"),
            Artifact(sha256=shas[1], size=6, name="y.bin"),
        ),
    )
    single = Artifact(sha256=shas[2], size=6, name="z.bin")
    cached = Artifact(sha256=shas[3], size=6, name="w.bin")
    seed = tmp_path / "seed.bin"  # already warm — must not re-download
    seed.write_bytes(kids[shas[3]])
    store._cache._write_blob(shas[3], seed)

    paths = store.get_many([(tree, tmp_path / "t"), (single, tmp_path / "z.bin"), (cached, tmp_path / "w.bin")])
    assert api.calls == 1  # one batched pull for everything missing
    assert api.downloads == 3  # the warm blob stayed local
    assert [p.name for p in paths] == ["t", "z.bin", "w.bin"]
    assert (tmp_path / "t" / "x.bin").read_bytes() == b"blob a"
    assert (tmp_path / "z.bin").read_bytes() == b"blob c"
    assert (tmp_path / "w.bin").read_bytes() == b"blob d"


def test_get_refs_resolves_present_and_absent_in_one_round_trip(tmp_path: Path):
    art = Artifact(sha256="a" * 64, size=3, name="m.json")
    refs = {
        "refs/exp/metrics.json": json.dumps(art.to_dict()).encode(),
        "refs/exp/arrays.json": json.dumps(art.to_dict()).encode(),
    }
    store, api = _store(tmp_path, refs)

    out = store.get_refs(["exp/metrics", "exp/arrays", "exp/unset"])
    assert out == {"exp/metrics": art, "exp/arrays": art, "exp/unset": None}
    assert api.calls == 1  # one download for the set; absence came from paths-info

    assert store.get_ref("exp/unset") is None  # the single-name path shares the batch machinery
    assert store.get_ref("exp/metrics") == art

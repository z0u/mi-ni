"""Tests for the content-addressed artifact store.

Two layers: the :class:`~mini.store.LocalStore` backend (hashing, idempotency,
trees, refs, publish) and the contextvar front door (``put``/``get`` resolving
the ambient store, the same shape as ``get_data_dir``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mini.store import (
    Artifact,
    LocalStore,
    get,
    get_ref,
    get_store,
    producer_context,
    publish,
    publish_repo,
    put,
    resolved_refs_context,
    set_ref,
    store_bucket,
    store_for,
    store_context,
    store_root_for,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalStore:
    return LocalStore(tmp_path / "store")


# ---------------------------------------------------------------------------
# Artifact handle
# ---------------------------------------------------------------------------


def test_content_type_inferred_from_name():
    assert Artifact("sha", 1, "fig.png").content_type == "image/png"
    assert Artifact("sha", 1, "data.json").content_type == "application/json"
    assert Artifact("sha", 1, "blob").content_type == "application/octet-stream"  # no extension


def test_content_type_explicit_overrides_guess():
    assert Artifact("sha", 1, "blob", media_type="image/svg+xml").content_type == "image/svg+xml"


def test_artifact_round_trips_through_dict():
    tree = Artifact("root", 6, "acts", kind="tree", children=(Artifact("a", 4, "a.bin"), Artifact("b", 2, "sub/b.bin")))
    assert Artifact.from_dict(tree.to_dict()) == tree


# ---------------------------------------------------------------------------
# LocalStore: blobs
# ---------------------------------------------------------------------------


def test_put_bytes_round_trips(store: LocalStore, tmp_path: Path):
    art = store.put(b"hello world", name="greeting.txt")
    assert art.kind == "file"
    assert store.has(art.sha256)
    out = store.get(art, tmp_path / "out.txt")
    assert out.read_bytes() == b"hello world"


def test_put_is_content_addressed_and_idempotent(store: LocalStore):
    """Identical bytes coincide regardless of name; a re-put writes no second blob."""
    a = store.put(b"same", name="one.bin")
    b = store.put(b"same", name="two.bin")  # different name, same bytes
    assert a.sha256 == b.sha256
    blobs = [p for p in (store.root / "cas").rglob("*") if p.is_file()]
    assert len(blobs) == 1  # stored once (under its two-char shard dir)


def test_blobs_are_sharded_by_prefix(store: LocalStore):
    """A blob lands under a two-char shard dir (``cas/ab/abcd…``), not a flat tree."""
    art = store.put(b"hello world", name="greeting.txt")
    assert (store.root / "cas" / art.sha256[:2] / art.sha256).is_file()


def test_put_file_hashes_streaming(store: LocalStore, tmp_path: Path):
    src = tmp_path / "data.bin"
    src.write_bytes(b"\x00\x01\x02" * 1000)
    art = store.put(src, name="data.bin")
    assert art.size == 3000
    assert store.get(art, tmp_path / "rt.bin").read_bytes() == src.read_bytes()


# ---------------------------------------------------------------------------
# LocalStore: trees
# ---------------------------------------------------------------------------


def test_put_get_tree_round_trips(store: LocalStore, tmp_path: Path):
    src = tmp_path / "acts"
    (src / "sub").mkdir(parents=True)
    (src / "layer0.bin").write_bytes(b"AAAA")
    (src / "sub" / "layer1.bin").write_bytes(b"BB")

    tree = store.put(src, name="acts")
    assert tree.kind == "tree"
    assert tree.size == 6
    assert {c.name for c in tree.children} == {"layer0.bin", "sub/layer1.bin"}

    dest = store.get(tree, tmp_path / "resolved")
    assert (dest / "layer0.bin").read_bytes() == b"AAAA"
    assert (dest / "sub" / "layer1.bin").read_bytes() == b"BB"


def test_identical_trees_share_blobs(store: LocalStore, tmp_path: Path):
    """A file shared between two trees is stored once (per-file dedup)."""
    for d in ("one", "two"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "shared.bin").write_bytes(b"SHARED")
    (tmp_path / "two" / "extra.bin").write_bytes(b"EXTRA")

    t1 = store.put(tmp_path / "one", name="one")
    t2 = store.put(tmp_path / "two", name="two")
    shared = {c.sha256 for c in t1.children} & {c.sha256 for c in t2.children}
    assert len(shared) == 1  # the shared file's blob coincides
    assert t1.sha256 != t2.sha256  # but the manifests differ


# ---------------------------------------------------------------------------
# LocalStore: refs + publish
# ---------------------------------------------------------------------------


def test_refs_round_trip_including_nested_names(store: LocalStore):
    art = store.put(b"corpus", name="corpus.bin")
    store.set_ref("datasets/tiny/v1", art)
    assert store.get_ref("datasets/tiny/v1") == art


def test_get_ref_missing_returns_none(store: LocalStore):
    assert store.get_ref("nope") is None


# ---------------------------------------------------------------------------
# Ref provenance: producer stamping + resolution tracking
# ---------------------------------------------------------------------------


def test_set_ref_stamps_the_ambient_producer(store: LocalStore):
    art = store.put(b"curves", name="curves.json")
    with producer_context({"experiment": "prep", "task": "abc123", "git_sha": "d" * 40}):
        store.set_ref("shared/curves", art)
    producer = store.ref_producer("shared/curves")
    assert producer is not None
    assert producer["experiment"] == "prep" and producer["task"] == "abc123"
    assert producer["written_at"]  # stamped at write time
    assert store.get_ref("shared/curves") == art  # the handle round-trips unchanged


def test_set_ref_outside_producer_context_is_unstamped(store: LocalStore):
    store.set_ref("shared/plain", store.put(b"x", name="x.bin"))
    assert store.ref_producer("shared/plain") is None
    assert store.get_ref("shared/plain") is not None


def test_get_ref_reads_pre_producer_payloads(store: LocalStore):
    # A ref written before producer stamping existed: the bare artifact dict.
    art = store.put(b"old", name="old.bin")
    store._write_ref("legacy/ref", json.dumps(art.to_dict(), sort_keys=True))
    assert store.get_ref("legacy/ref") == art
    assert store.ref_producer("legacy/ref") is None


def test_resolved_refs_context_collects_what_a_step_reads(store: LocalStore):
    art = store.put(b"a", name="a.bin")
    with producer_context({"experiment": "prep"}):
        store.set_ref("shared/a", art)
    store.set_ref("shared/anon", store.put(b"b", name="b.bin"))  # unstamped

    seen: dict[str, dict | None] = {}
    with resolved_refs_context(seen):
        store.get_ref("shared/a")
        store.get_ref("shared/anon")
        store.get_ref("shared/missing")  # unset — resolves to nothing, so no evidence
    assert seen["shared/a"] and seen["shared/a"]["experiment"] == "prep"
    assert seen["shared/anon"] is None  # read, but unattributable
    assert "shared/missing" not in seen


def test_publish_writes_extensioned_view_and_returns_url(store: LocalStore):
    art = store.put(b"\x89PNG fake", name="source.png")
    url = store.publish(art, "reports/exp/loss.png")
    assert url.startswith("file://")
    published = store.published / "reports" / "exp" / "loss.png"
    assert published.read_bytes() == b"\x89PNG fake"  # served by extension, not bare sha


def test_publish_rejects_trees(store: LocalStore, tmp_path: Path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f.bin").write_bytes(b"x")
    tree = store.put(tmp_path / "d", name="d")
    with pytest.raises(ValueError, match="single file"):
        store.publish(tree, "reports/x")


# ---------------------------------------------------------------------------
# Project scoping + ambient store
# ---------------------------------------------------------------------------


def test_store_root_is_project_scoped_beside_the_volume():
    # A volume at <root>/<experiment> shares <root>/store with every experiment.
    assert store_root_for(Path("/proj/.mini/acts")) == Path("/proj/.mini/store")


def test_store_bucket_reads_pyproject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pyproject.toml").write_text('[tool.mini]\nstore-bucket = "ns/bkt"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    assert store_bucket() == "ns/bkt"


def test_store_bucket_env_overrides_pyproject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pyproject.toml").write_text('[tool.mini]\nstore-bucket = "ns/bkt"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINI_STORE_BUCKET", "other/override")
    assert store_bucket() == "other/override"


def test_store_bucket_unset_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    assert store_bucket() is None


def test_publish_repo_reads_pyproject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pyproject.toml").write_text('[tool.mini]\npublish-repo = "ns/pub"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    assert publish_repo() == "ns/pub"


def test_publish_repo_env_overrides_pyproject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pyproject.toml").write_text('[tool.mini]\npublish-repo = "ns/pub"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINI_PUBLISH_REPO", "other/override")
    assert publish_repo() == "other/override"


def test_publish_repo_unset_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    assert publish_repo() is None


def test_store_for_threads_publish_repo_into_the_hfstore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mini.hf_store import HFStore

    monkeypatch.setenv("MINI_STORE_BUCKET", "ns/bkt")
    monkeypatch.setenv("MINI_PUBLISH_REPO", "ns/pub")
    monkeypatch.setattr("mini.store._hf_token", lambda: "tok")
    store = store_for(tmp_path / "store")
    assert isinstance(store, HFStore)
    assert store.publish_repo == "ns/pub"  # a bucket for the CAS, a repo for the publish tier


def test_store_for_falls_back_to_local_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A configured bucket but no HF token (trying the repo, pre-`./go auth`) → local, not a crash."""
    monkeypatch.setenv("MINI_STORE_BUCKET", "ns/bkt")
    monkeypatch.setattr("mini.store._hf_token", lambda: None)
    assert isinstance(store_for(tmp_path / "store"), LocalStore)


def test_store_for_builds_a_publish_only_store_from_a_repo_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A publish-repo with no bucket → a CAS-less HFStore for read-only export serving (the CI build)."""
    from mini.hf_store import HFStore

    monkeypatch.setattr("mini.store.store_bucket", lambda: None)  # no CAS bucket configured
    monkeypatch.setattr("mini.store.publish_repo", lambda: "ns/pub")
    monkeypatch.setattr("mini.store._hf_token", lambda: None)  # a public repo needs no token
    store = store_for(tmp_path / "store")
    assert isinstance(store, HFStore)
    assert store.bucket is None and store.publish_repo == "ns/pub"
    # It can still serve exports — that's the whole point (build reads exports off the repo).
    assert store.export_base("demo") == "https://huggingface.co/datasets/ns/pub/resolve/main/exports/demo/"


def test_publish_only_store_errors_on_cas_operations(tmp_path: Path):
    """A CAS-less store names the missing config rather than failing opaquely on the bucket call."""
    from mini.hf_store import HFStore

    store = HFStore(None, cache=LocalStore(tmp_path / "cache"), publish_repo="ns/pub")
    with pytest.raises(RuntimeError, match="no CAS bucket"):
        store.list_refs()


def test_store_for_uses_bucket_with_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mini.hf_store import HFStore

    monkeypatch.setenv("MINI_STORE_BUCKET", "ns/bkt")
    monkeypatch.setattr("mini.store._hf_token", lambda: "tok")
    store = store_for(tmp_path / "store")
    assert isinstance(store, HFStore)
    assert store._cache.root == tmp_path / "store-cache" / "hf"  # warm cache sits beside root by default


def test_store_for_cache_root_moves_the_warm_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A Modal worker points the warm cache at container-local disk, off the committed Volume."""
    from mini.hf_store import HFStore

    monkeypatch.setenv("MINI_STORE_BUCKET", "ns/bkt")
    monkeypatch.setattr("mini.store._hf_token", lambda: "tok")
    store = store_for(tmp_path / "vol" / "store", cache_root=tmp_path / "ephemeral")
    assert isinstance(store, HFStore)
    assert store._cache.root == tmp_path / "ephemeral"


def test_get_store_raises_outside_context():
    with pytest.raises(RuntimeError, match="No store configured"):
        get_store()


def test_ambient_put_get_and_refs(store: LocalStore, tmp_path: Path):
    with store_context(store):
        art = put(b"payload", name="p.bin")
        set_ref("k", art)
        assert get_ref("k") == art
        assert get(art, tmp_path / "p.bin").read_bytes() == b"payload"
        url = publish(art, "reports/p.bin")
        assert url.startswith("file://")
    with pytest.raises(RuntimeError):  # ambient store resets on exit
        get_store()

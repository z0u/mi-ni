"""Integration test for the Hugging Face bucket store — network-gated.

Talks to a real bucket, so it's skipped unless ``MINI_STORE_BUCKET`` and
``HF_TOKEN`` are set. It writes only under a unique ``cas/`` blob and a
per-run ``refs/_test/<uuid>`` / ``published/_test/<uuid>`` prefix, and deletes
everything it created in teardown, so it never collides with real artifacts.

Run it with::

    MINI_STORE_BUCKET=<ns>/<bucket> HF_TOKEN=... uv run pytest tests/mini/test_hf_store.py
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import pytest

from mini.store import _cas_key

BUCKET = os.environ.get("MINI_STORE_BUCKET")
PUBLISH_REPO = os.environ.get("MINI_PUBLISH_REPO")

pytestmark = pytest.mark.skipif(
    not (BUCKET and os.environ.get("HF_TOKEN")),
    reason="set MINI_STORE_BUCKET + HF_TOKEN to run the HF bucket integration test",
)

# The publish-tier cases also need a (public) dataset repo — see the split in #38.
repo_publish = pytest.mark.skipif(
    not (BUCKET and PUBLISH_REPO and os.environ.get("HF_TOKEN")),
    reason="also set MINI_PUBLISH_REPO to run the publish-repo integration test",
)


@pytest.fixture
def hf(tmp_path: Path):
    """An HFStore against the real bucket, with a unique prefix and full cleanup."""
    from huggingface_hub import HfApi

    from mini.hf_store import HFStore
    from mini.store import LocalStore

    assert BUCKET is not None  # narrowed by pytestmark skip
    tag = secrets.token_hex(4)
    store = HFStore(BUCKET, cache=LocalStore(tmp_path / "cache"))
    created: list[str] = []
    yield store, tag, created
    # Teardown: remove every path this test created.
    if created:
        HfApi(token=os.environ["HF_TOKEN"]).batch_bucket_files(BUCKET, delete=sorted(set(created)))


def test_put_get_round_trips_over_the_bucket(hf):
    store, tag, created = hf
    data = f"mini hf round-trip {tag}".encode()
    art = store.put(data, name="probe.txt")
    created.append(_cas_key(art.sha256))

    assert store.has(art.sha256)
    # Resolve through a *fresh* cache to force a real download, not a cache hit.
    from mini.hf_store import HFStore
    from mini.store import LocalStore

    fresh = HFStore(store.bucket, cache=LocalStore(Path(store._cache.root).parent / "cache2"))
    out = fresh.get(art, Path(store._cache.root).parent / "out.txt")
    assert out.read_bytes() == data


def test_ref_round_trips_over_the_bucket(hf):
    store, tag, created = hf
    art = store.put(f"ref payload {tag}".encode(), name="r.bin")
    created.append(_cas_key(art.sha256))
    name = f"_test/{tag}/handle"
    store.set_ref(name, art)
    created.append(f"refs/{name}.json")

    assert store.get_ref(name) == art
    assert store.get_ref(f"_test/{tag}/missing") is None


def test_publish_serves_with_content_type_from_extension(hf):
    store, tag, created = hf
    png = b"\x89PNG\r\n\x1a\n" + tag.encode()  # not a real PNG, but a .png name
    art = store.put(png, name="fig.png")
    created.append(_cas_key(art.sha256))
    path = f"_test/{tag}/fig.png"
    url = store.publish(art, path)
    created.append(f"published/{path}")

    assert url == f"https://huggingface.co/buckets/{BUCKET}/resolve/published/{path}"
    import requests

    head = requests.get(url, timeout=30)
    assert head.status_code == 200
    assert head.headers["content-type"].startswith("image/png")  # inferred from the extension


def test_export_round_trips_over_the_bucket(hf, tmp_path: Path):
    """A report bundle syncs as-is and fetches back — the publish→build handoff."""
    store, tag, created = hf
    key = f"_test/{tag}/report"
    src = tmp_path / "export"
    (src / "_assets").mkdir(parents=True)
    (src / "index.html").write_text(f'<img src="_assets/fig.png"> {tag}')
    (src / "_assets" / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n" + tag.encode())

    assert store.fetch_export(key, tmp_path / "miss") is False  # nothing synced yet
    store.sync_export(src, key)
    created += [f"exports/{key}/index.html", f"exports/{key}/_assets/fig.png"]

    dest = tmp_path / "pulled"
    assert store.fetch_export(key, dest) is True
    assert (dest / "index.html").read_text().endswith(tag)
    assert (dest / "_assets" / "fig.png").read_bytes().endswith(tag.encode())
    assert store.export_base(key) == f"https://huggingface.co/buckets/{BUCKET}/resolve/exports/{key}/"


# -- publish tier on a dataset repo (the private-CAS / public-publish split, #38) -----


@pytest.fixture
def hf_repo(tmp_path: Path):
    """An HFStore whose CAS is the bucket but whose publish tier is a dataset repo.

    Cleans up both sides: the ``cas/`` blobs it wrote to the bucket and the
    ``published/`` / ``exports/`` files it committed to the repo.
    """
    from huggingface_hub import HfApi

    from mini.hf_store import HFStore
    from mini.store import LocalStore

    assert BUCKET is not None and PUBLISH_REPO is not None  # narrowed by the repo_publish skip
    tag = secrets.token_hex(4)
    store = HFStore(BUCKET, cache=LocalStore(tmp_path / "cache"), publish_repo=PUBLISH_REPO)
    cas_created: list[str] = []
    repo_paths: list[str] = []
    yield store, tag, cas_created, repo_paths
    api = HfApi(token=os.environ["HF_TOKEN"])
    if cas_created:
        api.batch_bucket_files(BUCKET, delete=sorted(set(cas_created)))
    for p in sorted(set(repo_paths)):
        try:
            api.delete_file(path_in_repo=p, repo_id=PUBLISH_REPO, repo_type="dataset")
        except Exception:  # a test that failed before the upload left nothing to delete
            pass


@repo_publish
def test_publish_lands_on_the_dataset_repo(hf_repo):
    store, tag, cas_created, repo_paths = hf_repo
    png = b"\x89PNG\r\n\x1a\n" + tag.encode()
    art = store.put(png, name="fig.png")  # into the CAS bucket
    cas_created.append(_cas_key(art.sha256))
    path = f"_test/{tag}/fig.png"
    url = store.publish(art, path)  # copy-through into the public repo
    repo_paths.append(f"published/{path}")

    assert url == f"https://huggingface.co/datasets/{PUBLISH_REPO}/resolve/main/published/{path}"
    import requests

    r = requests.get(url, timeout=30)
    assert r.status_code == 200
    assert r.content == png  # the resolve URL serves the published bytes back


@repo_publish
def test_export_round_trips_over_the_repo(hf_repo, tmp_path: Path):
    store, tag, cas_created, repo_paths = hf_repo
    key = f"_test/{tag}/report"
    src = tmp_path / "export"
    (src / "_assets").mkdir(parents=True)
    (src / "index.html").write_text(f'<img src="_assets/fig.png"> {tag}')
    (src / "_assets" / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n" + tag.encode())

    assert store.fetch_export(key, tmp_path / "miss") is False  # nothing committed yet
    store.sync_export(src, key)
    repo_paths += [f"exports/{key}/index.html", f"exports/{key}/_assets/fig.png"]

    dest = tmp_path / "pulled"
    assert store.fetch_export(key, dest) is True
    assert (dest / "index.html").read_text().endswith(tag)
    assert (dest / "_assets" / "fig.png").read_bytes().endswith(tag.encode())
    assert store.export_base(key) == f"https://huggingface.co/datasets/{PUBLISH_REPO}/resolve/main/exports/{key}/"

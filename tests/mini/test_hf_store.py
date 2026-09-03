"""Integration test for the Hugging Face bucket store — network-gated.

Talks to a real bucket at ~2-3s per commit, so it is deselected by default (the ``hf`` marker) and, when selected, skipped unless a bucket is configured (``MINI_STORE_BUCKET``, else ``[tool.mini] store-bucket``) *and* the ambient Hugging Face token (``HF_TOKEN``, else the ``hf auth login`` cache) can write to it — a read-only token skips the module rather than failing every write. It writes only under a unique ``cas/`` blob and a per-run ``refs/_test/<uuid>`` / ``published/_test/<uuid>`` prefix, and deletes everything it created in teardown, so it never collides with real artifacts.

Run it with::

    ./go auth   # or set HF_TOKEN
    uv run pytest -m hf
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path

import pytest

from mini.store import _cas_key, _hf_token, publish_repo, store_bucket

BUCKET = store_bucket()
PUBLISH_REPO = publish_repo()
TOKEN = _hf_token()

pytestmark = [
    pytest.mark.hf,
    pytest.mark.skipif(
        not (BUCKET and TOKEN),
        reason="no HF bucket/token configured — run ./go auth to exercise the HF bucket integration test",
    ),
]

# The publish-tier cases also need a (public) dataset repo — see the split in #38.
repo_publish = pytest.mark.skipif(
    not PUBLISH_REPO, reason="no MINI_PUBLISH_REPO / publish-repo configured for the publish-repo integration test"
)

# Once a project has adopted the #38 split (MINI_PUBLISH_REPO set), the CAS bucket is
# expected to be private (see "Enabling it" in eng/publishing.md) — so a bucket-only
# publish() can no longer serve anonymously, and this case doesn't apply.
bucket_publish = pytest.mark.skipif(
    PUBLISH_REPO is not None,
    reason="MINI_PUBLISH_REPO is set — the CAS bucket is expected to be private (#38), "
    "so bucket-only publish() isn't publicly readable here",
)


@pytest.fixture(scope="module")
def api():
    """One authenticated client, after a write probe: a token that *exists* isn't one that can write.

    Claude Code web sessions carry a read-only token on purpose, and with an existence-only gate every write case there fails on a 403 that has nothing to do with the branch. The probe costs one round trip (a tiny ref written and deleted) and turns that into a skip that names the cause.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    assert BUCKET is not None  # narrowed by pytestmark skip
    api = HfApi(token=TOKEN)
    probe = f"refs/_test/probe-{secrets.token_hex(4)}.json"
    try:
        api.batch_bucket_files(BUCKET, add=[(b"{}", probe)])
    except HfHubHTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status in (401, 403):
            pytest.skip(
                f"the HF token can't write to {BUCKET} (HTTP {status}); the bucket integration test needs a write token"
            )
        raise
    api.batch_bucket_files(BUCKET, delete=[probe])
    return api


@pytest.fixture
def hf(api, tmp_path: Path):
    """An HFStore against the real bucket, with a unique prefix and full cleanup."""
    from mini.hf_store import HFStore
    from mini.store import LocalStore

    assert BUCKET is not None  # narrowed by pytestmark skip
    tag = secrets.token_hex(4)
    store = HFStore(BUCKET, cache=LocalStore(tmp_path / "cache"), token=TOKEN)
    created: list[str] = []
    yield store, tag, created
    # Teardown: remove every path this test created.
    if created:
        api.batch_bucket_files(BUCKET, delete=sorted(set(created)))


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


def test_batched_refs_and_gets_round_trip_over_the_bucket(hf, tmp_path: Path):
    """get_refs + get_many resolve a set in one pull — the report-loading fast path."""
    store, tag, created = hf
    arts = {}
    for i in range(3):
        art = store.put(f"batched {tag} {i}".encode(), name=f"b{i}.txt")
        created.append(_cas_key(art.sha256))
        name = f"_test/{tag}/batch/{i}"
        store.set_ref(name, art)
        created.append(f"refs/{name}.json")
        arts[name] = art

    from mini.hf_store import HFStore
    from mini.store import LocalStore

    fresh = HFStore(store.bucket, cache=LocalStore(tmp_path / "cold"))  # force real pulls
    names = [*arts, f"_test/{tag}/batch/missing"]
    resolved = fresh.get_refs(names)
    assert resolved == {**arts, f"_test/{tag}/batch/missing": None}

    items = [(art, tmp_path / f"out{i}.txt") for i, art in enumerate(arts.values())]
    outs = fresh.get_many(items)
    assert [p.read_bytes() for p in outs] == [f"batched {tag} {i}".encode() for i in range(3)]


def test_ref_round_trips_over_the_bucket(hf):
    store, tag, created = hf
    art = store.put(f"ref payload {tag}".encode(), name="r.bin")
    created.append(_cas_key(art.sha256))
    name = f"_test/{tag}/handle"
    store.set_ref(name, art)
    created.append(f"refs/{name}.json")

    assert store.get_ref(name) == art
    assert store.get_ref(f"_test/{tag}/missing") is None


@bucket_publish
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
    """A report bundle syncs as-is; the HTML reads back and the assets serve — the publish→build handoff."""
    store, tag, created = hf
    key = f"_test/{tag}/report"
    src = tmp_path / "export"
    (src / "_assets").mkdir(parents=True)
    (src / "index.html").write_text(f'<img src="_assets/fig.png"> {tag}')
    png = b"\x89PNG\r\n\x1a\n" + tag.encode()
    (src / "_assets" / "fig.png").write_bytes(png)

    assert store.read_export_html(key) is None  # nothing synced yet
    store.sync_export(src, key)
    created += [f"exports/{key}/index.html", f"exports/{key}/_assets/fig.png"]

    assert (store.read_export_html(key) or "").endswith(tag)
    assert store.export_base(key) == f"https://huggingface.co/buckets/{BUCKET}/resolve/exports/{key}/"
    # The read pulls only the HTML, but the sync still has to have put the whole bundle
    # up there — that's what the <base> points at. (Whether it serves *anonymously*
    # depends on the bucket being public; see bucket_publish.)
    assert store._remote_has(f"exports/{key}/_assets/fig.png")


# -- publish tier on a dataset repo (the private-CAS / public-publish split, #38) -----


@pytest.fixture
def hf_repo(api, tmp_path: Path):
    """An HFStore whose CAS is the bucket but whose publish tier is a dataset repo.

    Cleans up both sides: the ``cas/`` blobs it wrote to the bucket and the ``published/`` / ``exports/`` files it committed to the repo.
    """
    from mini.hf_store import HFStore
    from mini.store import LocalStore

    assert BUCKET is not None and PUBLISH_REPO is not None  # narrowed by the repo_publish skip
    tag = secrets.token_hex(4)
    store = HFStore(BUCKET, cache=LocalStore(tmp_path / "cache"), publish_repo=PUBLISH_REPO, token=TOKEN)
    cas_created: list[str] = []
    repo_paths: list[str] = []
    yield store, tag, cas_created, repo_paths
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

    # The URL pins to the commit the upload made — immutable, citable.
    assert re.fullmatch(
        f"https://huggingface.co/datasets/{PUBLISH_REPO}/resolve/[0-9a-f]{{40}}/published/{re.escape(path)}", url
    )
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

    assert store.read_export_html(key) is None  # nothing committed yet
    rev = store.sync_export(src, key)
    repo_paths += [f"exports/{key}/index.html", f"exports/{key}/_assets/fig.png"]
    assert rev is not None and re.fullmatch("[0-9a-f]{40}", rev)  # the revision a build pins to

    assert (store.read_export_html(key) or "").endswith(tag)
    assert store.export_base(key) == f"https://huggingface.co/datasets/{PUBLISH_REPO}/resolve/main/exports/{key}/"
    pinned_base = f"https://huggingface.co/datasets/{PUBLISH_REPO}/resolve/{rev}/exports/{key}/"
    assert store.export_base(key, revision=rev) == pinned_base
    # The build never copies the assets, so the <base> has to reach them where they sit.
    import requests

    r = requests.get(f"{pinned_base}_assets/fig.png", timeout=30)
    assert r.status_code == 200 and r.content.endswith(tag.encode())


@repo_publish
def test_pinned_export_survives_a_republish(hf_repo, tmp_path: Path):
    """The staging guarantee: overwriting ``exports/<key>/`` can't touch a pinned revision.

    This is what makes a pre-merge publish safe — production HTML is built against the pinned commit, so a branch re-publishing the same key swaps only the mutable head.
    """
    store, tag, cas_created, repo_paths = hf_repo
    key = f"_test/{tag}/report"
    src = tmp_path / "export"
    src.mkdir()
    (src / "index.html").write_text(f"v1 {tag}")
    rev1 = store.sync_export(src, key)
    repo_paths.append(f"exports/{key}/index.html")

    (src / "index.html").write_text(f"v2 {tag}")  # a branch re-publishes the same key
    rev2 = store.sync_export(src, key)
    assert rev1 != rev2

    assert store.read_export_html(key, revision=rev1) == f"v1 {tag}"  # the pin still serves v1
    assert store.read_export_html(key) == f"v2 {tag}"  # only the mutable head moved

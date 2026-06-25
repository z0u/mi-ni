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

BUCKET = os.environ.get('MINI_STORE_BUCKET')

pytestmark = pytest.mark.skipif(
    not (BUCKET and os.environ.get('HF_TOKEN')),
    reason='set MINI_STORE_BUCKET + HF_TOKEN to run the HF bucket integration test',
)


@pytest.fixture
def hf(tmp_path: Path):
    """An HFStore against the real bucket, with a unique prefix and full cleanup."""
    from huggingface_hub import HfApi

    from mini.hf_store import HFStore
    from mini.store import LocalStore

    assert BUCKET is not None  # narrowed by pytestmark skip
    tag = secrets.token_hex(4)
    store = HFStore(BUCKET, cache=LocalStore(tmp_path / 'cache'))
    created: list[str] = []
    yield store, tag, created
    # Teardown: remove every path this test created.
    if created:
        HfApi(token=os.environ['HF_TOKEN']).batch_bucket_files(BUCKET, delete=sorted(set(created)))


def test_put_get_round_trips_over_the_bucket(hf):
    store, tag, created = hf
    data = f'mini hf round-trip {tag}'.encode()
    art = store.put(data, name='probe.txt')
    created.append(f'cas/{art.sha256}')

    assert store.has(art.sha256)
    # Resolve through a *fresh* cache to force a real download, not a cache hit.
    from mini.hf_store import HFStore
    from mini.store import LocalStore

    fresh = HFStore(BUCKET, cache=LocalStore(Path(store._cache.root).parent / 'cache2'))
    out = fresh.get(art, Path(store._cache.root).parent / 'out.txt')
    assert out.read_bytes() == data


def test_ref_round_trips_over_the_bucket(hf):
    store, tag, created = hf
    art = store.put(f'ref payload {tag}'.encode(), name='r.bin')
    created.append(f'cas/{art.sha256}')
    name = f'_test/{tag}/handle'
    store.set_ref(name, art)
    created.append(f'refs/{name}.json')

    assert store.get_ref(name) == art
    assert store.get_ref(f'_test/{tag}/missing') is None


def test_publish_serves_with_content_type_from_extension(hf):
    store, tag, created = hf
    png = b'\x89PNG\r\n\x1a\n' + tag.encode()  # not a real PNG, but a .png name
    art = store.put(png, name='fig.png')
    created.append(f'cas/{art.sha256}')
    path = f'_test/{tag}/fig.png'
    url = store.publish(art, path)
    created.append(f'published/{path}')

    assert url == f'https://huggingface.co/buckets/{BUCKET}/resolve/published/{path}'
    import requests

    head = requests.get(url, timeout=30)
    assert head.status_code == 200
    assert head.headers['content-type'].startswith('image/png')  # inferred from the extension

"""Tests for the apparatus module."""

import asyncio
import contextlib
import time
from pathlib import Path
from typing import cast

import pytest

from mini.progress import emit_progress
from mini.volume import get_data_dir
from mini.local_apparatus import LocalApparatus
from mini.modal_apparatus import ModalApparatus
import modal


# ---------------------------------------------------------------------------
# Mock Modal App — simulates Modal's behaviour so we can test ModalApparatus
# without network access.
# ---------------------------------------------------------------------------


class _MockModalMap:
    """Simulates Modal's map interface (sync + async)."""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *input_iterators, kwargs=None, order_outputs=True, return_exceptions=False):
        del order_outputs, return_exceptions
        kw = kwargs or {}
        for args in zip(*input_iterators, strict=False):
            yield self._fn(*args, **kw)

    async def aio(self, *input_iterators, kwargs=None, order_outputs=True, return_exceptions=False):
        del order_outputs, return_exceptions
        kw = kwargs or {}
        for args in zip(*input_iterators, strict=False):
            yield self._fn(*args, **kw)


class _MockModalFunction:
    """Simulates ``modal.Function`` produced by ``@app.function()``."""

    def __init__(self, fn):
        self._fn = fn
        self.map = _MockModalMap(fn)


class _AsyncNoop:
    """Callable that returns a no-op coroutine. Used to mock Modal's .aio interface."""

    async def __call__(self, *args, **kwargs):
        pass


class MockModalImage:
    """Simulates ``modal.Image`` for testing."""

    class build:
        """Mock build that supports both sync and async (.aio) calls."""

        aio = _AsyncNoop()

        def __init__(self, app):
            del app


class MockModalQueue:
    """Simulates ``modal.Queue`` for testing."""

    def __init__(self):
        self._items = []

    def put(self, item, block=True, timeout=None):
        del block, timeout
        self._items.append(item)

    def get_many(self, batch_size, block=True, timeout=None):
        del block, timeout
        result = self._items[:batch_size]
        self._items = self._items[batch_size:]
        return result

    def len(self):
        return len(self._items)

    @staticmethod
    @contextlib.asynccontextmanager
    async def ephemeral():
        """Return a mock ephemeral queue."""
        yield MockModalQueue()


class MockModalVolume:
    """Simulates ``modal.Volume`` for testing."""

    def commit(self):
        """Mock commit — no-op for testing."""
        pass


class MockModalApp:
    """Simulates ``modal.App`` for testing."""

    def __init__(self, name: str = "test"):
        self.name = name
        self.app_id = "mock-app-id"  # Add app_id for newer Modal versions  # noqa
        self.function_kwargs: dict = {}

    def function(self, **decorator_kwargs):
        self.function_kwargs = decorator_kwargs

        def decorator(fn):
            return _MockModalFunction(fn)

        return decorator

    def run(self):
        """Return an async context manager (no-op)."""
        return contextlib.AsyncExitStack()


# ---------------------------------------------------------------------------
# Fixtures — each test runs against both apparatus
# ---------------------------------------------------------------------------


def _make_local():
    return LocalApparatus("test", max_workers=1)


def _make_modal(monkeypatch):
    monkeypatch.setattr("modal.Queue", MockModalQueue)
    monkeypatch.setattr("modal.enable_output", contextlib.nullcontext)
    monkeypatch.setattr("modal.Volume.from_name", lambda name, create_if_missing=False: MockModalVolume())  # noqa
    app = ModalApparatus(cast(modal.App, MockModalApp()))
    # Provide a mock image to avoid real Modal API calls in tests
    app.modal_fn_kwargs["image"] = MockModalImage()
    return app


@pytest.fixture(params=["local", "modal"], ids=["LocalApparatus", "ModalApparatus"])
def apparatus(request, monkeypatch):
    if request.param == "local":
        return _make_local()
    return _make_modal(monkeypatch)


# ---------------------------------------------------------------------------
# Parameter-passing tests — both apparatus must behave identically
# ---------------------------------------------------------------------------


def test_single_arg(apparatus):
    """map(fn, [a, b, c]) calls fn(a), fn(b), fn(c)."""
    results = list(apparatus.map(lambda x: x * 2, [1, 2, 3]))
    assert results == [2, 4, 6]


def test_two_args(apparatus):
    """map(fn, xs, ys) calls fn(x, y) for each pair."""
    results = list(apparatus.map(lambda x, y: f"{x}-{y}", [1, 2, 3], ["a", "b", "c"]))
    assert results == ["1-a", "2-b", "3-c"]


def test_single_arg_with_kwargs(apparatus):
    """map(fn, xs, kwargs={...}) forwards kwargs to every call."""

    def fn(x, scale=1):
        return x * scale

    results = list(apparatus.map(fn, [1, 2, 3], kwargs={"scale": 10}))
    assert results == [10, 20, 30]


def test_two_args_with_kwargs(apparatus):
    """map(fn, xs, ys, kwargs={...}) forwards both positional and keyword args."""

    def fn(x, y, sep=","):
        return f"{x}{sep}{y}"

    results = list(apparatus.map(fn, [1, 2], ["a", "b"], kwargs={"sep": ":"}))
    assert results == ["1:a", "2:b"]


def test_kwargs_only(apparatus):
    """map(fn, dummy_iter, kwargs={...}) works with functions that only use kwargs."""

    def fn(_, key="default"):
        return key

    results = list(apparatus.map(fn, range(3), kwargs={"key": "hello"}))
    assert results == ["hello", "hello", "hello"]


def test_no_kwargs(apparatus):
    """map(fn, xs) works without kwargs (kwargs defaults to None)."""

    def fn(x, y="default"):
        return f"{x}-{y}"

    results = list(apparatus.map(fn, [1, 2]))
    assert results == ["1-default", "2-default"]


def test_empty(apparatus):
    """map with empty iterables returns no results."""
    results = list(apparatus.map(lambda x: x, []))
    assert results == []


def test_result_order_preserved(apparatus):
    """Results are returned in the same order as inputs, not completion order."""
    results = list(apparatus.map(lambda x: x**2, [3, 1, 4, 1, 5]))
    assert results == [9, 1, 16, 1, 25]


def test_amap_materializes(apparatus):
    """amap yields results that can be materialized in async contexts."""

    async def collect():
        return [result async for result in apparatus.amap(lambda x: x + 1, [1, 2, 3])]

    results = asyncio.run(collect())
    assert results == [2, 3, 4]


def test_modal_auth_error_has_actionable_message(monkeypatch):
    """Modal auth errors are re-raised with a concise remediation hint."""
    app = _make_modal(monkeypatch)

    async def broken_amap(*args, **kwargs):
        del args, kwargs
        raise modal.exception.AuthError("not authenticated")
        # pyrefly: ignore [unreachable]
        yield  # pragma: no cover

    monkeypatch.setattr(app, "_amap", broken_amap)

    async def collect():
        return [result async for result in app.amap(lambda x: x, [1])]

    with pytest.raises(RuntimeError, match=r"Modal authentication failed\. Run \./go auth, then try again\."):
        asyncio.run(collect())


def test_memo_worker_mounts_hf_cache(monkeypatch):
    """The remote worker gets the shared HF cache Volume, with HF_HOME pointing at it.

    That's what lets a multi-stage pipeline's ``from_pretrained`` reuse weights
    across containers instead of re-downloading per container (#50).
    """
    from mini.modal_apparatus import HF_CACHE_MOUNT

    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    secrets_made: list[dict] = []
    monkeypatch.setattr("modal.Secret.from_dict", lambda d: secrets_made.append(d) or ("secret", d))
    app = _make_modal(monkeypatch)
    app._memo_worker()
    kwargs = app.app.function_kwargs  # pyrefly: ignore [missing-attribute]  (MockModalApp)
    assert isinstance(kwargs["volumes"][HF_CACHE_MOUNT], MockModalVolume)
    assert {"HF_HOME": HF_CACHE_MOUNT} in secrets_made


def test_attach_hf_cache_preserves_user_mounts_and_secrets(monkeypatch):
    from mini.modal_apparatus import HF_CACHE_MOUNT, _attach_hf_cache

    monkeypatch.setattr("modal.Volume.from_name", lambda name, create_if_missing=False: MockModalVolume())
    monkeypatch.setattr("modal.Secret.from_dict", lambda d: ("secret", d))
    fn_kwargs = {"volumes": {"/vol": "user-vol"}, "secrets": ["user-secret"]}
    _attach_hf_cache(fn_kwargs)
    assert fn_kwargs["volumes"].keys() == {"/vol", HF_CACHE_MOUNT}
    assert fn_kwargs["secrets"] == ["user-secret", ("secret", {"HF_HOME": HF_CACHE_MOUNT})]


def test_complex_objects_as_args(apparatus):
    """map works with non-trivial argument types (dicts, dataclasses, etc.)."""

    def fn(params):
        return params["a"] + params["b"]

    results = list(apparatus.map(fn, [{"a": 1, "b": 2}, {"a": 10, "b": 20}]))
    assert results == [3, 30]


# ---------------------------------------------------------------------------
# LocalApparatus-specific tests
# ---------------------------------------------------------------------------


def test_local_apparatus_concurrent():
    """LocalApparatus with multiple workers runs concurrently."""
    app = LocalApparatus("test", max_workers=3)
    start = time.monotonic()

    def slow(x):
        time.sleep(0.1)
        return x

    results = list(app.map(slow, [1, 2, 3]))
    elapsed = time.monotonic() - start
    assert results == [1, 2, 3]
    assert elapsed < 0.25


def test_local_apparatus_progress_emission():
    """Mapped functions can emit progress messages."""
    app = LocalApparatus("test", max_workers=1)

    def fn_with_progress(x):
        for i in range(10):
            emit_progress(i, 10, message=f"step {i}")
        return x

    results = list(app.map(fn_with_progress, [1, 2]))
    assert results == [1, 2]


def test_progress_emission_outside_apparatus():
    """emit_progress() silently does nothing when not inside a run context."""
    # Should not raise an exception
    emit_progress(0, 10, message="test")


def test_local_apparatus_exception_propagates():
    """Exceptions in mapped functions propagate to the caller."""
    app = LocalApparatus("test", max_workers=1)

    def fail(x):
        if x == 2:
            raise ValueError("bad value")
        return x

    results = []
    try:
        for r in app.map(fail, [1, 2, 3]):
            results.append(r)
    except ValueError:
        pass
    assert results == [1]


# ---------------------------------------------------------------------------
# Volume integration tests — both apparatus must provide get_data_dir()
# ---------------------------------------------------------------------------


def test_get_data_dir_available_in_mapped_function(apparatus):
    """get_data_dir() returns a Path inside a mapped function."""

    def fn(x):
        d = get_data_dir()
        assert isinstance(d, Path)
        return d

    results = list(apparatus.map(fn, [1, 2]))
    assert len(results) == 2
    assert all(isinstance(r, Path) for r in results)


# ---------------------------------------------------------------------------
# ModalRecordStore — the memo control plane on a modal.Dict. A plain dict
# satisfies the same get/keys/__setitem__ surface, so we test the contract
# without the network.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ambient artifact store — the interactive (map/arun) path must bind it too,
# not only the detached memo worker (issue #39).
# ---------------------------------------------------------------------------


def test_interactive_local_map_resolves_ambient_store(tmp_path: Path, monkeypatch):
    """A fn mapped via LocalApparatus (not the memo worker) can put/get artifacts;
    the blob lands under the ``store/`` root sibling to the experiment's data dir."""
    from mini.store import get, put

    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)  # force a LocalStore
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    app = LocalApparatus("exp", data_dir=tmp_path / "exp")

    def fn(x):
        art = put(f"blob-{x}".encode(), name=f"{x}.bin")
        return get(art, tmp_path / f"out-{x}.bin").read_bytes()

    assert list(app.map(fn, [1, 2])) == [b"blob-1", b"blob-2"]
    blobs = [p for p in (tmp_path / "store" / "cas").rglob("*") if p.is_file()]
    assert len(blobs) == 2  # rooted beside the data dir, not under it


def test_wrap_for_modal_binds_store_under_data_dir(tmp_path: Path, monkeypatch):
    """The Modal-wrapped fn binds an ambient store rooted at ``data_dir/store`` —
    under the mounted Volume, since the parent isn't shared remotely."""
    from mini.local_queue import LocalQueue
    from mini.modal_apparatus import _wrap_for_modal
    from mini.store import LocalStore, get_store

    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)  # force a LocalStore
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)

    def fn():
        store = get_store()
        assert isinstance(store, LocalStore)
        return store.root

    wrapped = _wrap_for_modal(fn, [], "run", queue=LocalQueue(), kwargs={}, emission_interval=1.0, data_dir=tmp_path)
    assert wrapped(0) == tmp_path / "store"


def test_modal_record_store_contract():
    from mini.modal_apparatus import ModalRecordStore

    store = ModalRecordStore({})
    assert store.read("k") is None
    store.write("k", {"key": "k", "state": "running"})
    assert store.read("k") == {"key": "k", "state": "running"}
    store.merge("k", {"step": 3})  # merge preserves existing fields
    assert store.read("k") == {"key": "k", "state": "running", "step": 3}
    store.write("k", {"key": "k", "state": "running"})  # write resets wholesale
    assert store.read("k") == {"key": "k", "state": "running"}
    assert store.keys() == ["k"]


class _FakeModalDict(dict):
    """A dict with ``modal.Dict``'s insert-if-absent verb (`put(skip_if_exists=)`)."""

    def put(self, key, value, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


def test_modal_write_if_claims_fresh_key_via_insert_if_absent():
    """The double-spawn race on a never-run key resolves atomically: the claim
    goes through ``put(skip_if_exists=True)``, so the second ticker loses even
    with no compare-and-swap."""
    from mini.modal_apparatus import ModalRecordStore

    store = ModalRecordStore(_FakeModalDict())
    assert store.write_if("k", {"key": "k", "gen": "a"}, None) is True
    assert store.write_if("k", {"key": "k", "gen": "b"}, None) is False  # already claimed
    assert store.read("k") == {"key": "k", "gen": "a"}


def test_modal_write_if_reclaims_reset_record():
    """A reset record (present but unclaimed) defeats insert-if-absent, so the
    claim falls through to read-check-write — and still lands."""
    from mini.modal_apparatus import ModalRecordStore

    store = ModalRecordStore(_FakeModalDict({"k": {"key": "k", "state": None}}))
    assert store.write_if("k", {"key": "k", "gen": "a"}, None) is True
    assert store.write_if("k", {"key": "k", "gen": "c"}, "b") is False  # fenced: wrong gen
    assert store.write_if("k", {"key": "k", "gen": "c"}, "a") is True  # supersede gen a

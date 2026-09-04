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
from modal.call_graph import InputStatus
from modal.exception import FunctionTimeoutError, InternalFailure, NotFoundError, OutputExpiredError
from modal.exception import TimeoutError as ModalTimeout


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


@pytest.mark.parametrize(
    ("fn", "iterables", "kwargs", "expected"),
    [
        pytest.param(lambda x: x * 2, ([1, 2, 3],), None, [2, 4, 6], id="one-iterable"),
        pytest.param(
            lambda x, y: f"{x}-{y}",
            ([1, 2, 3], ["a", "b", "c"]),
            None,
            ["1-a", "2-b", "3-c"],
            id="iterables-zipped",
        ),
        pytest.param(lambda x, scale=1: x * scale, ([1, 2, 3],), {"scale": 10}, [10, 20, 30], id="kwargs-forwarded"),
    ],
)
def test_map_passes_arguments_through(apparatus, fn, iterables, kwargs, expected):
    """map zips the iterables into positional args, in order, and forwards ``kwargs`` to every call."""
    assert list(apparatus.map(fn, *iterables, kwargs=kwargs)) == expected


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
        yield  # noqa: V201  # pragma: no cover  — the yield is what makes this an async generator

    monkeypatch.setattr(app, "_amap", broken_amap)

    async def collect():
        return [result async for result in app.amap(lambda x: x, [1])]

    with pytest.raises(RuntimeError, match=r"Modal authentication failed\. Run \./go auth, then try again\."):
        asyncio.run(collect())


def test_memo_worker_mounts_hf_cache(monkeypatch):
    """The remote worker gets the shared HF cache Volume, with HF_HOME pointing at it.

    That's what lets a multi-stage pipeline's ``from_pretrained`` reuse weights across containers instead of re-downloading per container (#50).
    """
    from mini.modal_apparatus import HF_CACHE_MOUNT

    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    monkeypatch.setenv("MINI_NO_PROJECT_CONFIG", "1")  # and the pyproject / mini.local.toml fallback, workers included
    secrets_made: list[dict] = []
    monkeypatch.setattr("modal.Secret.from_dict", lambda d: secrets_made.append(d) or ("secret", d))

    def train_step(x):
        return x

    app = _make_modal(monkeypatch)
    app._memo_worker(train_step)  # one registered worker per task fn (named after it)
    kwargs = app.app.function_kwargs  # pyrefly: ignore [missing-attribute]  (MockModalApp)
    assert isinstance(kwargs["volumes"][HF_CACHE_MOUNT], MockModalVolume)
    assert {"HF_HOME": HF_CACHE_MOUNT} in secrets_made
    assert kwargs["name"].startswith("train_step-")  # dashboard shows the task fn, not _modal_task_entry


def test_attach_hf_cache_preserves_user_mounts_and_secrets(monkeypatch):
    from mini.modal_apparatus import HF_CACHE_MOUNT, _attach_hf_cache

    monkeypatch.setattr("modal.Volume.from_name", lambda name, create_if_missing=False: MockModalVolume())
    monkeypatch.setattr("modal.Secret.from_dict", lambda d: ("secret", d))
    fn_kwargs = {"volumes": {"/vol": "user-vol"}, "secrets": ["user-secret"]}
    _attach_hf_cache(fn_kwargs)
    assert fn_kwargs["volumes"].keys() == {"/vol", HF_CACHE_MOUNT}
    assert fn_kwargs["secrets"] == ["user-secret", ("secret", {"HF_HOME": HF_CACHE_MOUNT})]


# ---------------------------------------------------------------------------
# LocalApparatus-specific tests
# ---------------------------------------------------------------------------


def test_local_apparatus_concurrent():
    """LocalApparatus with multiple workers runs concurrently.

    Measured as overlap rather than as elapsed wall clock. A total-time bound reads the machine as much as the pool: on a loaded shared container the same three sleeps took ~1.9s against a 0.25s bound, so the test failed on a pristine tree. Every worker is a thread here, and ``sleep`` releases the GIL, so what ``max_workers=3`` actually promises is that all three spans are open at once — which is what this asserts, at any speed.
    """
    app = LocalApparatus("test", max_workers=3)

    def slow(x):
        started = time.monotonic()
        time.sleep(0.2)  # margin over thread start-up, which is microseconds even under contention
        return x, started, time.monotonic()

    results = list(app.map(slow, [1, 2, 3]))
    assert [x for x, _, _ in results] == [1, 2, 3]
    assert max(s for _, s, _ in results) < min(e for _, _, e in results)  # all three spans intersect


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
    with pytest.raises(ValueError, match="bad value"):
        for r in app.map(fail, [1, 2, 3]):
            results.append(r)
    assert results == [1]  # the results before the failure still reached the caller


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


def test_interactive_local_map_resolves_ambient_store(tmp_path: Path, local_store):
    """A fn mapped via LocalApparatus (not the memo worker) can put/get artifacts; the blob lands under the ``store/`` root sibling to the experiment's data dir."""
    from mini.store import get, put

    app = LocalApparatus("exp", data_dir=tmp_path / "exp")

    def fn(x):
        art = put(f"blob-{x}".encode(), name=f"{x}.bin")
        return get(art, tmp_path / f"out-{x}.bin").read_bytes()

    assert list(app.map(fn, [1, 2])) == [b"blob-1", b"blob-2"]
    blobs = [p for p in (tmp_path / "store" / "cas").rglob("*") if p.is_file()]
    assert len(blobs) == 2  # rooted beside the data dir, not under it


def test_wrap_for_modal_binds_store_under_data_dir(tmp_path: Path, local_store):
    """The Modal-wrapped fn binds an ambient store rooted at ``data_dir/store`` — under the mounted Volume, since the parent isn't shared remotely."""
    from mini.local_queue import LocalQueue
    from mini.modal_apparatus import _wrap_for_modal
    from mini.store import LocalStore, get_store

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
    """The double-spawn race on a never-run key resolves atomically: the claim goes through ``put(skip_if_exists=True)``, so the second ticker loses even with no compare-and-swap."""
    from mini.modal_apparatus import ModalRecordStore

    store = ModalRecordStore(_FakeModalDict())
    assert store.write_if("k", {"key": "k", "gen": "a"}, None) is True
    assert store.write_if("k", {"key": "k", "gen": "b"}, None) is False  # already claimed
    assert store.read("k") == {"key": "k", "gen": "a"}


def test_modal_write_if_reclaims_reset_record():
    """A reset record (present but unclaimed) defeats insert-if-absent, so the claim falls through to read-check-write — and still lands."""
    from mini.modal_apparatus import ModalRecordStore

    store = ModalRecordStore(_FakeModalDict({"k": {"key": "k", "state": None}}))
    assert store.write_if("k", {"key": "k", "gen": "a"}, None) is True
    assert store.write_if("k", {"key": "k", "gen": "c"}, "b") is False  # fenced: wrong gen
    assert store.write_if("k", {"key": "k", "gen": "c"}, "a") is True  # supersede gen a


class _CountingModalDict(_FakeModalDict):
    """A fake Dict that records each round-trip, so a test can count them."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ops: list[str] = []

    def get(self, key, default=None):
        self.ops.append("get")
        return super().get(key, default)

    def __setitem__(self, key, value) -> None:
        self.ops.append("set")
        super().__setitem__(key, value)


def test_modal_merge_if_costs_one_round_trip_each_way():
    """Every progress update from every worker lands through ``merge_if``, and on Modal each step of it is a network call. The inherited version reads to check the fence and then reads again for something to merge onto — three round-trips where one read serves both. A fenced-out merge writes nothing at all."""
    from mini.modal_apparatus import ModalRecordStore

    d = _CountingModalDict({"k": {"key": "k", "gen": "a", "state": "running"}})
    store = ModalRecordStore(d)

    d.ops.clear()
    assert store.merge_if("k", {"step": 3}, "a") is True
    assert d.ops == ["get", "set"]
    assert store.read("k") == {"key": "k", "gen": "a", "state": "running", "step": 3}  # merged, not replaced

    d.ops.clear()
    assert store.merge_if("k", {"step": 4}, "b") is False  # superseded — this worker no longer owns the record
    assert d.ops == ["get"]  # …and it costs one read to find out, with no write
    assert (store.read("k") or {})["step"] == 3


# ---------------------------------------------------------------------------
# Modal liveness probe — reap_dead's _is_task_alive. A *settled* failure
# (function timeout, terminated, init failure) must read dead, or a killed
# worker's record shows RUNNING forever (z0u/sca2#20); ambiguity stays alive
# (a false "dead" would double-spawn a live GPU task on retry).
# ---------------------------------------------------------------------------


class _FakeFunctionCall:
    """A ``modal.FunctionCall`` whose probe endpoints we script per-test."""

    object_id = "fc-under-test"

    def __init__(self, get_exc=None, graph=None):
        self._get_exc, self._graph = get_exc, graph

    def get(self, timeout=None):
        if self._get_exc is not None:
            raise self._get_exc
        return None

    def get_call_graph(self):
        if isinstance(self._graph, Exception):
            raise self._graph
        return self._graph or []


def _graph_input(status, function_call_id="fc-under-test"):
    from modal.call_graph import InputInfo

    return InputInfo("in-1", function_call_id, "ta-1", status, "worker", "mod", [])


_LAUNCHED = {"fc_id": "fc-under-test"}


@pytest.mark.parametrize(
    ("record", "fake", "alive"),
    [
        # Not launched on Modal yet — nothing to probe, never reap:
        pytest.param({}, None, True, id="no-fc-id"),
        # Direct signals out of ``get(timeout=0)``:
        pytest.param(_LAUNCHED, _FakeFunctionCall(), True, id="returned"),  # the record settles on its own
        # Poll came up empty — the *builtin*, as modal 1.5.1 actually raises, and modal's own, for good measure:
        pytest.param(_LAUNCHED, _FakeFunctionCall(TimeoutError()), True, id="builtin-timeout"),
        pytest.param(_LAUNCHED, _FakeFunctionCall(ModalTimeout()), True, id="modal-timeout"),
        pytest.param(_LAUNCHED, _FakeFunctionCall(FunctionTimeoutError("timeout")), False, id="function-timeout"),
        pytest.param(_LAUNCHED, _FakeFunctionCall(InternalFailure("infra")), False, id="internal-failure"),
        pytest.param(_LAUNCHED, _FakeFunctionCall(OutputExpiredError()), False, id="output-expired"),
        pytest.param(_LAUNCHED, _FakeFunctionCall(NotFoundError("gone")), False, id="not-found"),
        # Ambiguous exception → the call-graph cross-check discriminates:
        pytest.param(
            _LAUNCHED,
            _FakeFunctionCall(RuntimeError("deserialized remote failure"), [_graph_input(InputStatus.FAILURE)]),
            False,
            id="graph-failure",
        ),
        pytest.param(
            _LAUNCHED,
            _FakeFunctionCall(RuntimeError("worker terminated"), [_graph_input(InputStatus.TERMINATED)]),
            False,
            id="graph-terminated",
        ),
        pytest.param(
            _LAUNCHED,
            _FakeFunctionCall(RuntimeError("transport blip"), [_graph_input(InputStatus.PENDING)]),
            True,
            id="graph-pending",
        ),
        pytest.param(
            _LAUNCHED,
            _FakeFunctionCall(RuntimeError("transport blip"), [_graph_input(InputStatus.SUCCESS)]),
            True,
            id="graph-success",
        ),
        pytest.param(
            _LAUNCHED,
            _FakeFunctionCall(RuntimeError("transport down"), RuntimeError("graph unreachable too")),
            True,
            id="graph-unreachable",
        ),
        # Another call's failed input in the graph is not evidence about ours:
        pytest.param(
            _LAUNCHED,
            _FakeFunctionCall(RuntimeError("blip"), [_graph_input(InputStatus.FAILURE, function_call_id="fc-other")]),
            True,
            id="other-calls-failure",
        ),
    ],
)
def test_liveness_settled_states(monkeypatch, record, fake, alive):
    """A settled failure must read dead; anything ambiguous stays alive."""
    monkeypatch.setattr("modal.FunctionCall.from_id", lambda fc_id: fake)
    assert _make_modal(monkeypatch)._is_task_alive(record) is alive


def test_reap_settles_timeout_killed_modal_task(monkeypatch, tmp_path):
    """End to end: a timeout-killed call's RUNNING record settles FAILED on reap, so ``status``/``watch`` can't read it as running forever (z0u/sca2#20)."""
    from mini.memo import MemoStore

    store = MemoStore(tmp_path / "exp")
    store.records_backend.merge("t1", {"key": "t1", "state": "running", "gen": "g1", "fc_id": "fc-under-test"})
    monkeypatch.setattr("modal.FunctionCall.from_id", lambda fc_id: _FakeFunctionCall(FunctionTimeoutError("t")))
    app = _make_modal(monkeypatch)
    assert app.reap_dead(store) == ["t1"]
    rec = store.record("t1")
    assert rec["state"] == "failed" and rec["gen"] is None and "vanished" in rec["error"]


# ---------------------------------------------------------------------------
# Worker environment (`env=`)
# ---------------------------------------------------------------------------


def test_modal_env_becomes_a_container_secret():
    """``env=`` reaches the container as a Secret, and never as a ``@function`` kwarg — Modal has no ``env`` parameter, so a leak would be a TypeError at registration. The container is the right level: a task that set it in-process would be too late for anything that reads its env once at init.

    The HF store/cache secrets are attached alongside, so ``env`` must append rather than replace; an absent or empty mapping adds nothing at all.
    """
    from mini.modal_apparatus import _attach_env

    fn_kwargs: dict = {"gpu": "L4", "env": {"XLA_FLAGS": "--xla_gpu_deterministic_ops=true"}}
    _attach_env(fn_kwargs)
    assert "env" not in fn_kwargs
    assert fn_kwargs["gpu"] == "L4"
    assert len(fn_kwargs["secrets"]) == 1
    assert isinstance(fn_kwargs["secrets"][0], modal.Secret)

    sentinel = modal.Secret.from_dict({"EXISTING": "1"})
    existing: dict = {"secrets": [sentinel], "env": {"XLA_FLAGS": "-x"}}
    _attach_env(existing)
    assert existing["secrets"][0] is sentinel and len(existing["secrets"]) == 2

    for empty in ({}, None):
        bare: dict = {"gpu": "L4"} | ({"env": empty} if empty is not None else {})
        _attach_env(bare)
        assert bare == {"gpu": "L4"}


def test_modal_w_merges_env_key_by_key(monkeypatch):
    """Unlike every other option, ``env`` merges: a role adds one variable without having to restate the project-wide defaults it inherits."""
    app = _make_modal(monkeypatch).w(env={"XLA_FLAGS": "-x", "KEEP": "1"}, gpu="T4")
    role = app.w(env={"XLA_FLAGS": "-y"}, gpu="L4")
    assert role.modal_fn_kwargs["env"] == {"XLA_FLAGS": "-y", "KEEP": "1"}
    assert role.modal_fn_kwargs["gpu"] == "L4"  # everything else still replaces
    assert app.modal_fn_kwargs["env"] == {"XLA_FLAGS": "-x", "KEEP": "1"}  # caller untouched


def test_local_env_reaches_the_task_worker_subprocess(tmp_path, monkeypatch):
    """Locally the memoized path is a subprocess per task, so it can carry the same env. ``.w(env=)`` merges like Modal's, and the value lands in the child's environment rather than only the parent's."""
    captured: dict = {}

    def fake_popen(argv, **kwargs):
        captured.update(kwargs)
        return type("P", (), {"pid": 4321})()

    monkeypatch.setattr("mini.runs.subprocess.Popen", fake_popen)
    from mini.runs import spawn_taskworker

    app = LocalApparatus("envexp", data_dir=tmp_path / "envexp").w(env={"XLA_FLAGS": "-x"}).w(env={"OTHER": "2"})
    assert app.env == {"XLA_FLAGS": "-x", "OTHER": "2"}

    assert spawn_taskworker(tmp_path, "k", env=app.env) == 4321
    assert captured["env"]["XLA_FLAGS"] == "-x" and captured["env"]["OTHER"] == "2"
    assert "PATH" in captured["env"]  # overlaid on the parent env, not a replacement

    captured.clear()
    spawn_taskworker(tmp_path, "k")  # no env → inherit, unchanged from before
    assert captured["env"] is None

"""
Apparatus for running sweeps on Modal infrastructure.

Example::

    from mini.modal_apparatus import ModalApparatus

    app = ModalApparatus("my-experiment").w(gpu="T4", timeout=3600)
    results = list(app.map(train, configs))
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager, nullcontext
from functools import wraps
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterable, TypeVar, override

if TYPE_CHECKING:
    from mini.gc import GcIO

import cloudpickle
import modal

from mini._queues import QueueLike
from mini._tls import ensure_grpc_trusts_system_ca
from mini.apparatus import Apparatus
from mini.memo import MemoStore, RecordStore
from mini.modal_queue import ModalQueue
from mini.modal_volume import ModalVolume
from mini.progress import ProgressMessage, progress_context
from mini.progress_display import RichProgressDisplay
from mini.requirements import project_packages, uv_freeze
from mini.runs import data_root
from mini.store import (
    PUBLISH_REPO_ENV,
    STORE_BUCKET_ENV,
    Artifact,
    LocalStore,
    Store,
    _cas_key,
    _hf_token,
    publish_repo,
    store_bucket,
    store_context,
    store_for,
)
from mini.volume import data_dir_context

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["ModalApparatus", "ModalRecordStore", "ModalMemoStore", "control_dict_name"]

STARTUP_TIMEOUT_SECONDS = 120


def control_dict_name(name: str) -> str:
    """Name of the control-plane ``modal.Dict`` for experiment *name*.

    Module-level so a client can address the control plane without constructing
    a ``ModalApparatus`` (e.g. the CLI's other-backend peek on an empty read).
    """
    return f"mini-cp-{name}"


def _modal_auth_error_message() -> str:
    """Build a user-facing message for Modal authentication failures."""
    return "Modal authentication failed. Run ./go auth, then try again."


def _app_page_url(app: modal.App) -> str | None:
    """Extract the dashboard URL from a running Modal app.

    The URL comes from the backend via ``RunningApp.app_page_url``, but
    synchronicity hides ``_running_app`` on the public ``App`` wrapper.
    Reach through the wrapper to get it.
    """
    # synchronicity stores the underlying _App as the sole entry in __dict__
    inner_values = list(app.__dict__.values())
    if not inner_values:
        return None
    inner_app = inner_values[0]
    running = getattr(inner_app, "_running_app", None)
    if running is None:
        return None
    return running.app_page_url


def make_image() -> modal.Image:
    """Helper to create a Modal image with experiment dependencies.

    Includes the `cuda` dependency group (e.g. `jax[cuda12]`), which is excluded
    from local installs: locally we run CPU-only, while the remote image gets
    the CUDA plugin and picks up the GPU when one is attached.
    """
    deps = uv_freeze(all_groups=True, not_groups=["local", "dev"])
    project_deps = project_packages()
    print(f"Creating Modal image with dependencies: Project: {project_deps}")
    return (
        modal.Image.debian_slim()
        .pip_install(*deps)
        .add_local_python_source(*project_deps)
    )  # fmt: skip


# The Hugging Face cache tier: one workspace-wide Volume, mounted where HF_HOME
# points, so a multi-stage pipeline's `from_pretrained` pulls a model once — not
# once per container (each container otherwise boots with an empty $HF_HOME).
# One env var covers both HF sub-caches: `hub/` (model/dataset snapshots) and
# `xet/` (transfer chunks, size-capped by hf_xet's own default). Purely a
# disposable read accelerator, distinct from the artifact Store (durable,
# content-addressed) and the per-experiment Volume (working dir + checkpoints):
# no commit discipline (background commits suffice; a lost write costs one
# re-download), concurrent writers at worst pull the same model twice, and
# deleting the Volume is always safe. Locally this tier doesn't exist —
# ~/.cache/huggingface already persists.
HF_CACHE_VOLUME = "mini-hf-cache"
HF_CACHE_MOUNT = "/hf-cache"

# Container-local root for the worker's HFStore warm cache: deliberately NOT under
# the mounted Volume, where it would be committed alongside results and shadow
# every bucket artifact on paid storage. Ephemeral is correct — the bucket is the
# durable copy, and the cache only needs to outlive the container's own re-reads.
WORKER_STORE_CACHE = Path("/tmp/mini-store-cache")


def _attach_hf_cache(fn_kwargs: dict[str, Any]) -> None:
    """Mount the shared HF cache Volume and point ``HF_HOME`` at it (in place).

    The env var rides in a Secret rather than on the image so a user-supplied
    ``.w(image=...)`` still gets it.
    """
    cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
    fn_kwargs["volumes"] = {**fn_kwargs.get("volumes", {}), HF_CACHE_MOUNT: cache}
    fn_kwargs["secrets"] = [*fn_kwargs.get("secrets", []), modal.Secret.from_dict({"HF_HOME": HF_CACHE_MOUNT})]


# ---------------------------------------------------------------------------
# Memoized-orchestration backend (control plane = modal.Dict, I/O plane = Volume)
# ---------------------------------------------------------------------------


class ModalRecordStore(RecordStore):
    """``RecordStore`` backed by a named ``modal.Dict``.

    The Dict is readable/writable from the client with no remote function and no
    commit (Redis-backed), so polling never spins up compute. The same named
    Dict is opened by the remote worker to write back state/metrics. Records are
    tiny and last-writer-wins, so a Dict value per key is the natural fit.
    """

    def __init__(self, d: Any):
        # *d* is a ``modal.Dict`` (or any get/keys/__setitem__ mapping — a plain
        # dict for tests). Injected rather than opened here so the store is
        # testable without the network; use ``from_name`` for the real thing.
        self._d = d

    @classmethod
    def from_name(cls, name: str) -> ModalRecordStore:
        return cls(modal.Dict.from_name(name, create_if_missing=True))

    def read(self, key: str) -> dict[str, Any] | None:
        return self._d.get(key)

    def write(self, key: str, record: dict[str, Any]) -> None:
        self._d[key] = record

    def merge(self, key: str, fields: dict[str, Any]) -> None:
        cur = self._d.get(key) or {}
        cur.update(fields)
        self._d[key] = cur

    def keys(self) -> list[str]:
        return list(self._d.keys())

    def delete(self, key: str) -> None:
        try:
            self._d.pop(key)
        except KeyError:
            pass

    def write_if(self, key: str, record: dict[str, Any], gen: str | None) -> bool:
        """Gen-fenced write; the *first claim of a fresh key* is exact on Modal.

        ``modal.Dict`` has no compare-and-swap, but it does have insert-if-absent
        (``put(..., skip_if_exists=True)``) — which is exactly the shape of the
        double-spawn race two tickers run when they both classify a never-run key
        as launchable (``gen=None``, no record). Claiming through it makes that
        race lose atomically. The other transitions — re-claiming a reset record
        (present but unclaimed) or superseding gen *x* — have no matching
        primitive, so they stay read-check-write with a tiny window.
        """
        if gen is None and (put := getattr(self._d, "put", None)) is not None:
            if put(key, record, skip_if_exists=True):
                return True
            cur = self.read(key)
            if cur is None or cur.get("gen") is not None:
                return False  # claimed (or deleted and re-claimed) in between — lose
            # Present but unclaimed (a reset record awaiting re-run): the insert
            # can't take it, so fall through to the read-check-write path below.
        return super().write_if(key, record, gen)


class ModalMemoStore(MemoStore):
    """A ``MemoStore`` whose records live in a ``modal.Dict`` and whose results
    are read back from the Modal Volume (the remote worker writes them there and
    commits). Only the I/O-plane *reads* differ from the local store — the remote
    worker, with the Volume mounted, writes through a plain ``MemoStore``.
    """

    def __init__(self, volume: ModalVolume, records: RecordStore):
        super().__init__(volume.path, records=records)
        self._volume = volume

    def _read_volume_bytes(self, rel: str) -> bytes:
        # Client-side reads already reflect the worker's committed writes;
        # ``reload()`` is only valid inside a running function, not here.
        return b"".join(self._volume._modal_volume.read_file(rel))

    def result(self, key: str) -> Any:
        gen = self._gen(key)
        name = f"result-{gen}.pkl" if gen else "result.pkl"
        return cloudpickle.loads(self._read_volume_bytes(f"_memo/{key}/{name}"))

    def error(self, key: str) -> str:
        gen = self._gen(key)
        for name in dict.fromkeys((f"error-{gen}.txt" if gen else "error.txt", "error.txt")):
            try:
                return self._read_volume_bytes(f"_memo/{key}/{name}").decode()
            except FileNotFoundError, modal.exception.NotFoundError:
                continue
        return "(no logs)"

    def result_artifacts(self, key: str) -> list[str] | None:
        gen = self._gen(key)
        name = f"result-{gen}.artifacts.json" if gen else "result.artifacts.json"
        try:
            return json.loads(self._read_volume_bytes(f"_memo/{key}/{name}"))
        except FileNotFoundError, modal.exception.NotFoundError:
            return None  # pre-sidecar record: only unpickling the result reveals its refs


class ModalVolumeStore(Store):
    """Client-side artifact :class:`~mini.store.Store` that reads blobs off the Volume.

    The remote worker writes the CAS *under* the mounted Volume (``store/cas/ab/<sha>``)
    and commits it; this reads those blobs back from the client (a report, or
    ``ctx.run`` resolving a handle into the next step) with no running function,
    caching each blob into a local :class:`~mini.store.LocalStore` so a re-read is
    free. Read-only: producing artifacts happens *in* a step (on the worker), and a
    report that wants to publish a Modal-produced asset resolves it here, then
    ``put``/``publish``es through the local project store.
    """

    def __init__(self, volume: ModalVolume, cache: Any):
        self._volume = volume
        self._cache = cache  # a LocalStore checkout cache (warm copies, keyed by sha)

    def _read_volume_bytes(self, rel: str) -> bytes:
        return b"".join(self._volume._modal_volume.read_file(rel))

    def has(self, sha256: str) -> bool:
        if self._cache.has(sha256):
            return True
        try:
            self._read_volume_bytes(f"store/{_cas_key(sha256)}")
            return True
        except FileNotFoundError, modal.exception.NotFoundError:
            return False

    def _read_blob(self, sha256: str, dest: Path) -> None:
        import shutil

        blob = self._cache._blob_path(sha256)
        if not blob.exists():  # pull once into the warm cache, then serve locally
            data = self._read_volume_bytes(f"store/{_cas_key(sha256)}")
            blob.parent.mkdir(parents=True, exist_ok=True)
            tmp = blob.with_name(f"{sha256}.tmp")
            tmp.write_bytes(data)
            tmp.replace(blob)
        shutil.copyfile(blob, dest)

    def _read_ref(self, name: str) -> str | None:
        try:
            return self._read_volume_bytes(f"store/refs/{name}.json").decode()
        except FileNotFoundError, modal.exception.NotFoundError:
            return None

    def _write_blob(self, sha256: str, src: Path) -> None:
        raise NotImplementedError("ModalVolumeStore is read-only on the client; put() runs inside a step")

    def _write_ref(self, name: str, payload: str) -> None:
        raise NotImplementedError("ModalVolumeStore is read-only on the client; set_ref() runs inside a step")

    def publish(self, art: Artifact, path: str) -> str:
        raise NotImplementedError(
            "publish a Modal-produced asset by resolving it (get) into a local path, "
            "then put()/publish() through the local project store"
        )


def _hf_store_secret() -> modal.Secret | None:
    """A Modal Secret carrying the HF bucket config into the worker, if configured.

    When a bucket is configured (``[tool.mini] store-bucket`` or the env override)
    and a token is available, forward both into the remote container's env so the
    worker's ``store_for`` resolves to the shared bucket. Absent either, the
    worker falls back to the Volume-backed store.
    """
    bucket = store_bucket()
    if not bucket:
        return None
    token = _hf_token()  # env, or the cached `hf auth login`
    if not token:
        return None
    env: dict[str, str | None] = {STORE_BUCKET_ENV: bucket, "HF_TOKEN": token}
    if repo := publish_repo():  # so an in-step publish() targets the same tier as the driver
        env[PUBLISH_REPO_ENV] = repo
    return modal.Secret.from_dict(env)


def _modal_task_entry(blob: bytes, key: str, gen: str, dict_name: str, volume_name: str, mount_point: str) -> None:
    """Remote entry: run one memoized call on Modal and persist its result/state.

    Mirrors the local subprocess worker (``mini._taskworker``) but reads the call
    from the ``spawn`` argument (not disk), writes records to the ``modal.Dict``,
    and commits the Volume before flipping the record to a settled state.
    """
    import cloudpickle as _cp
    import modal as _modal

    from mini._taskworker import execute_task
    from mini.memo import MemoStore
    from mini.modal_apparatus import WORKER_STORE_CACHE, ModalRecordStore
    from mini.store import store_for

    fn, args, hooks = _cp.loads(blob)
    store = MemoStore(Path(mount_point), records=ModalRecordStore.from_name(dict_name))
    volume = _modal.Volume.from_name(volume_name)
    # With MINI_STORE_BUCKET set (passed in via a Modal Secret), put/get hit the
    # shared HF bucket — so another experiment, local or remote, reads these bytes
    # back with no shared Volume; the warm cache goes to container-local disk, not
    # the committed Volume (see WORKER_STORE_CACHE). Otherwise the CAS rides
    # *under* the mounted Volume (committed with the result), per-experiment until
    # #22's project Volume.
    artifacts = store_for(Path(mount_point) / "store", cache_root=WORKER_STORE_CACHE)
    execute_task(store, key, fn, args, hooks, commit=volume.commit, artifacts=artifacts, gen=gen)


class ModalApparatus(Apparatus[ModalVolume]):
    """
    Run functions on Modal.

    Usage::

        app = ModalApparatus("my-experiment").w(gpu="T4", timeout=3600)
        results = list(app.map(train, configs))
    """

    app: modal.App

    def __init__(self, app: modal.App | str):
        ensure_grpc_trusts_system_ca()  # work behind TLS-inspecting proxies (see mini._tls)
        if isinstance(app, str):
            name = app
            self.app = modal.App(name)
        else:
            if not app.name:
                raise ValueError("ModalApparatus requires a named modal.App")
            name = app.name
            self.app = app
        self.modal_fn_kwargs: dict[str, Any] = {
            # Don't let Modal silently retry failures — surface them immediately.
            "retries": 0,
        }
        # `max_containers` is deliberately *not* a global default: the detached memo
        # path wants to fan out (unbounded unless `--max-containers`/`.w()` caps it),
        # while the blocking `amap` path defaults to 1 — applied in `_build_modal_fn`.
        self._before_hooks: list[Callable[[], Any]] = []
        self._volume: ModalVolume | None = ModalVolume(name)
        self._memo_fn: modal.Function | None = None
        self._image: modal.Image | None = None  # lazily built default; see _ensure_image

    def __str__(self) -> str:
        return f'Modal apparatus "{self.app.name}"'

    def clone(self) -> ModalApparatus:
        new_app = ModalApparatus(self.app)
        new_app.modal_fn_kwargs = self.modal_fn_kwargs.copy()
        new_app._before_hooks = self._before_hooks[:]
        new_app._volume = self._volume
        new_app._image = self._image  # carry the cached image so a clone doesn't rebuild
        return new_app

    def _ensure_image(self) -> modal.Image:
        """The image for remote functions: a user override via ``.w(image=)`` if
        present, else the project default, built once and cached.

        Deferred (not built in ``__init__``) so read-only commands — ``status`` /
        ``results`` / ``cancel``, which only touch the ``modal.Dict`` and Volume —
        never run ``make_image`` (no ``uv`` freeze, no "Creating Modal image" noise).
        """
        if "image" in self.modal_fn_kwargs:
            return self.modal_fn_kwargs["image"]
        if self._image is None:
            self._image = make_image()
        return self._image

    @property
    def _dict_name(self) -> str:
        """Name of the control-plane ``modal.Dict`` for this experiment."""
        assert self.app.name  # guaranteed by __init__ (a named App is required)
        return control_dict_name(self.app.name)

    def w(self, **kwargs: Any) -> ModalApparatus:
        """
        Return a new apparatus with additional Modal function kwargs merged in.

        These kwargs are passed to the ``@app.function()`` decorator when
        mapping, and can be used to specify things like GPU requirements or
        timeouts.
        """
        new_app = self.clone()
        new_app.modal_fn_kwargs = {**self.modal_fn_kwargs, **kwargs}
        return new_app

    @override
    def before_each(self, hook: Callable[[], Any]) -> ModalApparatus:
        new_app = self.clone()
        new_app._before_hooks = self._before_hooks + [hook]
        return new_app

    # -- Memoized orchestration (detached) ------------------------------------

    @override
    def memo_store(self) -> MemoStore:
        return ModalMemoStore(self.volume, ModalRecordStore.from_name(self._dict_name))

    @override
    def gc_io(self, store: MemoStore) -> GcIO:
        from mini.gc import ModalGcIO

        return ModalGcIO(self.volume._modal_volume)

    @override
    def store(self) -> Store:
        """The artifact store for reads on the client (reports, ``ctx`` resolves).

        With a bucket configured *and* a token available, that's the shared HF
        bucket the worker wrote to — so artifacts read back the same everywhere, no
        Volume needed. Otherwise it's a read-through over this experiment's Modal
        Volume, warm-caching into a local checkout (``.mini/store-cache/<app>``).
        The token gate mirrors ``_hf_store_secret``: with no token the worker writes
        to the Volume, so the client must read from there too (not an empty bucket).
        """
        if store_bucket() and _hf_token():
            return store_for(data_root() / "store")
        assert self.app.name  # guaranteed by __init__ (a named App is required)
        cache = LocalStore(data_root() / "store-cache" / self.app.name)
        return ModalVolumeStore(self.volume, cache)

    def _memo_worker(self) -> modal.Function:
        """Register (once) and return the generic remote worker for memo tasks.

        One stable function serves every task; each spawned call carries its own
        cloudpickled call. The Volume is mounted so the worker writes results to
        the same path the client reads back from.

        A detached sweep should parallelise, so there's *no* ``max_containers``
        default here — it's unbounded unless the caller sets one
        (``--max-containers`` / ``.w(max_containers=N)``), which now passes through
        to cap concurrency/cost. Only ``startup_timeout`` (a client-side knob, not a
        ``@function`` kwarg) is dropped.
        """
        if self._memo_fn is None:
            drop = {"startup_timeout"}
            fn_kwargs = {k: v for k, v in self.modal_fn_kwargs.items() if k not in drop}
            fn_kwargs["image"] = self._ensure_image()
            if isinstance(self._volume, ModalVolume):
                fn_kwargs["volumes"] = {
                    **fn_kwargs.get("volumes", {}),
                    str(self._volume.path): self._volume._modal_volume,
                }
            if secret := _hf_store_secret():  # forward HF bucket creds so put/get hit the shared store
                fn_kwargs["secrets"] = [*fn_kwargs.get("secrets", []), secret]
            _attach_hf_cache(fn_kwargs)  # shared HF_HOME, so from_pretrained caches across containers
            self._memo_fn = self.app.function(serialized=True, **fn_kwargs)(_modal_task_entry)
        return self._memo_fn

    @override
    def spawn_tasks(self, store: MemoStore, batch: list[tuple[str, str, Callable, tuple, list]]) -> None:
        worker = self._memo_worker()
        dict_name, volume_name, mount_point = self._dict_name, self.app.name, str(self.volume.path)
        # One detached app context for the whole batch (the cost we batch away is
        # app setup/registration, not the per-task spawn), then one ``spawn`` per
        # task. Unlike ``spawn_map`` — which returns None on Modal 1.3.x — ``spawn``
        # yields a FunctionCall id per task, recorded immediately so a launch
        # failure is diagnosable (cross-check via FunctionCall.from_id, find logs
        # on the dashboard) even before the worker writes its first heartbeat.
        # max_containers is dropped in ``_memo_worker``, so the tasks parallelise.
        fc_ids: dict[str, tuple[str, str]] = {}
        with self.app.run(detach=True):
            for key, gen, fn, args, hooks in batch:
                blob = cloudpickle.dumps((fn, args, hooks))
                fc = worker.spawn(blob, key, gen, dict_name, volume_name, mount_point)
                fc_ids[key] = (gen, fc.object_id)
        now = time.time()
        for key, (gen, fc_id) in fc_ids.items():
            store.update_if(key, gen, fc_id=fc_id, heartbeat_at=now)

    @override
    def _stop_task(self, rec: dict[str, Any]) -> None:
        """Cancel the task's Modal ``FunctionCall`` by its recorded id."""
        from contextlib import suppress

        if fc_id := rec.get("fc_id"):
            with suppress(Exception):  # already finished / unknown id
                modal.FunctionCall.from_id(fc_id).cancel()

    @override
    def _is_task_alive(self, rec: dict[str, Any]) -> bool:
        """Probe the task's ``FunctionCall`` for liveness (for ``reap_dead``).

        ``get(timeout=0)`` polls without waiting: a ``modal.exception.TimeoutError``
        means the input is still unfinished (running/queued → alive); a normal
        return means it completed (the record settles on its own → treat as alive).
        Only the *definitive gone* signals — the output expired, or the call id is
        unknown — count as dead. We deliberately treat any other error (a remote
        infra failure, or a transient read-side network blip) as alive: a false
        "dead" would mark a live GPU task FAILED, and a retry would double-spawn it.
        """
        fc_id = rec.get("fc_id")
        if not fc_id:
            return True  # not launched on Modal yet — nothing to probe
        from modal.exception import NotFoundError, OutputExpiredError
        from modal.exception import TimeoutError as ModalTimeout

        try:
            modal.FunctionCall.from_id(fc_id).get(timeout=0)
        except ModalTimeout:
            return True  # output not ready, input still unfinished → running/queued
        except OutputExpiredError, NotFoundError:
            return False  # the call is gone — it will never settle the record
        except Exception:
            return True  # ambiguous infra/network error — don't risk reaping a live task
        return True  # completed and returned → the record settles on its own

    @override
    async def amap(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ):
        try:
            async for result in self._amap(fn, *iterables, kwargs=kwargs):
                yield result
        except modal.exception.AuthError:
            log.debug("Modal authentication failed", exc_info=True)
            raise RuntimeError(_modal_auth_error_message()) from None

    async def _amap(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ):
        # TODO: support lazy iterables
        iterables_lists: list[list] = [list(it) for it in iterables]
        n = len(iterables_lists[0]) if iterables_lists else 0
        if n == 0:
            return

        log.info("Running %d jobs on Modal", n)
        run_id = secrets.token_hex(4)

        image: modal.Image = self._ensure_image()
        with modal.enable_output():
            async with self.app.run():
                await image.build.aio(self.app)

        async with modal.Queue.ephemeral() as progress_queue:
            display = RichProgressDisplay(total_jobs=n, queue=ModalQueue(progress_queue))
            modal_fn, startup_timeout = self._build_modal_fn(
                fn,
                run_id,
                display,
                kwargs=kwargs,
            )

            async with display, self.app.run():
                if url := _app_page_url(self.app):
                    print(f"View app at {url}")
                async with _startup_watchdog(display, startup_timeout):
                    async for result in modal_fn.map.aio(count(), *iterables_lists):
                        yield result

    def _build_modal_fn(
        self,
        fn: Callable[..., R],
        run_id: str,
        display: RichProgressDisplay,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[modal.Function, float]:
        """Wrap *fn* for Modal and register it with the app.

        Return ``(modal_function, startup_timeout)``.
        """
        # The blocking/interactive path defaults to one container (the memo path is
        # unbounded); set it here now that it's no longer a global default.
        max_containers = self.modal_fn_kwargs.get("max_containers", 1)
        emission_interval = max_containers / 10.0
        wrapped_fn = _wrap_for_modal(
            fn,
            self._before_hooks,
            run_id,
            queue=display.queue,
            kwargs=kwargs or {},
            emission_interval=emission_interval,
            data_dir=self._volume.path if self._volume is not None else None,
            commit_volume=(self._volume._modal_volume if isinstance(self._volume, ModalVolume) else None),
        )
        fn_kwargs: dict[str, Any] = {**self.modal_fn_kwargs}
        fn_kwargs.setdefault("max_containers", 1)  # interactive default; memo path stays unbounded
        fn_kwargs["image"] = self._ensure_image()
        startup_timeout: float = fn_kwargs.pop("startup_timeout", STARTUP_TIMEOUT_SECONDS)
        if isinstance(self._volume, ModalVolume):
            volumes = fn_kwargs.get("volumes", {})
            fn_kwargs["volumes"] = {
                **volumes,
                str(self._volume.path): self._volume._modal_volume,
            }
        if secret := _hf_store_secret():  # forward HF bucket creds so put/get hit the shared store
            fn_kwargs["secrets"] = [*fn_kwargs.get("secrets", []), secret]
        _attach_hf_cache(fn_kwargs)  # shared HF_HOME, so from_pretrained caches across containers
        modal_fn = self.app.function(serialized=True, **fn_kwargs)(wrapped_fn)
        return modal_fn, startup_timeout


@asynccontextmanager
async def _startup_watchdog(
    display: RichProgressDisplay,
    timeout_seconds: float,
) -> AsyncIterator[None]:
    """Raise if no remote container checks in within *timeout_seconds*.

    Once the display receives any message (set via ``display._any_message``),
    the deadline is cancelled and the body runs without a time limit.
    """
    try:
        async with asyncio.timeout(timeout_seconds) as scope:

            async def _cancel_on_first_message() -> None:
                await asyncio.to_thread(
                    display._any_message.wait,
                    timeout_seconds + 10,
                )
                scope.reschedule(None)

            watcher = asyncio.create_task(_cancel_on_first_message())
            try:
                yield
            finally:
                watcher.cancel()
    except TimeoutError:
        raise RuntimeError(
            f"No containers started within {timeout_seconds}s. "
            "Containers may be crash-looping — "
            "check the Modal dashboard for logs."
        ) from None


def _wrap_for_modal(
    fn: Callable[..., R],
    hooks: list[Callable[[], None]],
    run_id: str,
    queue: QueueLike[ProgressMessage],
    kwargs: dict[str, Any],
    emission_interval: float,
    data_dir: Path | None,
    commit_volume: modal.Volume | None = None,
) -> Callable[..., R]:
    @wraps(fn)
    def wrapped_fn(index: int, *args) -> R:
        # Signal that this container started successfully. Emitted directly
        # (not via the debouncer) so the caller-side watchdog sees it ASAP.
        queue.put(
            ProgressMessage(
                run_id=run_id,
                job_id=str(index),
                step=0,
                total=0,
                message="started",
            )
        )
        dir_ctx = data_dir_context(data_dir) if data_dir is not None else nullcontext()
        # Build the store remotely (this fn is serialized): the CAS rides under the
        # mounted Volume, whose parent isn't shared remotely — so no store_root_for.
        # The bucket path's warm cache stays off the committed Volume (WORKER_STORE_CACHE).
        store_ctx = (
            store_context(store_for(data_dir / "store", cache_root=WORKER_STORE_CACHE))
            if data_dir is not None
            else nullcontext()
        )
        with (
            progress_context(run_id, str(index), queue=queue, emission_interval=emission_interval),
            dir_ctx,
            store_ctx,
        ):
            for hook in reversed(hooks):
                hook()
            result = fn(*args, **kwargs)
            if commit_volume is not None:
                commit_volume.commit()
            return result

    # Give the wrapper a unique name so that repeated submissions of the same
    # function on a single App don't trigger Modal's name-collision warning.
    wrapped_fn.__name__ = f"{wrapped_fn.__name__}_{run_id}"
    wrapped_fn.__qualname__ = f"{wrapped_fn.__qualname__}_{run_id}"

    return wrapped_fn

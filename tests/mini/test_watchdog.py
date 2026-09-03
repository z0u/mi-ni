"""The wedged-worker liveness guard: progress watchdog + staleness surfacing.

A wedged worker (hung device call, deadlocked thread) holds its resources while making no step progress — and can keep emitting heartbeats, so heartbeat staleness never trips. The watchdog aborts it from inside (FAILED + stack dump
+ hard exit); the record carries the progress/heartbeat split so monitors can
tell "dead" from "slow"; per-key cancel reaps one wedge without stopping its healthy siblings.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

from mini import _watchdog
from mini._taskworker import _MemoSink, _phase_hook
from mini._watchdog import Watchdog
from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus
from mini.memo import MemoStore
from mini.orchestration import tick
from mini.progress import ProgressMessage, progress_context
from mini.runs import RunState, progress_age, stale_heartbeat, stale_progress
from mini.store import LocalStore


# ---------------------------------------------------------------------------
# Watchdog unit behavior
# ---------------------------------------------------------------------------


class _FakeClock:
    """Stands in for the ``time`` module inside :mod:`mini._watchdog`."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def monotonic(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """Fake the watchdog's *judgement* of elapsed time, not its polling.

    ``_run`` still waits on its Event at the real poll interval, so the thread behaves exactly as in production; only ``time.monotonic`` is ours. That way a test crosses a threshold by moving the clock instead of by sleeping through it, and costs a poll or two rather than the whole timeout.
    """
    fake = _FakeClock()
    monkeypatch.setattr(_watchdog, "time", fake)
    return fake


def _settle(exits: list[int], timeout_s: float = 5.0) -> None:
    """Wait (in real seconds) for the watchdog thread's next poll to land."""
    deadline = time.monotonic() + timeout_s
    while not exits and time.monotonic() < deadline:
        time.sleep(0.01)


def test_watchdog_fires_only_when_progress_stalls(clock: _FakeClock):
    stalls: list[str] = []
    exits: list[int] = []
    wd = Watchdog(0.4, stalls.append, _exit=exits.append)
    with wd:
        for step in range(3):  # steady progress: each poke resets the clock
            wd.poke(step, 3)
            clock.t += 0.3  # a gap under the timeout, so no poll ever sees a stall
            time.sleep(0.1)  # …and polls do happen: the interval here is 0.1s
        assert stalls == []
    assert exits == []


def test_watchdog_aborts_on_frozen_step(clock: _FakeClock):
    stalls: list[str] = []
    exits: list[int] = []
    wd = Watchdog(0.3, stalls.append, _exit=exits.append)
    with wd:
        wd.poke(7, 100)
        wd.poke(7, 100)  # emissions keep coming, but the step never advances
        clock.t += 1.0
        _settle(exits)
    assert exits == [70]
    (diagnosis,) = stalls
    assert "at step 7/100" in diagnosis
    assert "WatchdogStall" in diagnosis.strip().splitlines()[-1]
    assert "--- thread" in diagnosis  # the stack dump is the wedge's "traceback"


def test_grace_covers_setup_then_tight_timeout_applies(clock: _FakeClock):
    stalls: list[str] = []
    exits: list[int] = []
    wd = Watchdog(0.2, stalls.append, grace_s=5.0, _exit=exits.append)
    with wd:
        clock.t += 0.6  # "tokenizing": way past the 0.2s timeout, inside the grace
        time.sleep(0.1)  # a couple of polls, none of which finds anything wrong
        assert exits == []
        wd.poke(1, 10)  # first emission ends the grace; the tight timeout takes over
        clock.t += 0.5
        _settle(exits)
    assert exits == [70]
    assert "watchdog 0.2s" in stalls[0]  # the tight threshold, not the 5s grace


def test_phase_covers_a_step_free_span_then_the_tight_timeout_returns(clock: _FakeClock):
    stalls: list[str] = []
    exits: list[int] = []
    wd = Watchdog(0.2, stalls.append, _exit=exits.append)
    with wd:
        wd.poke(1, 10)  # training under way; the tight 0.2s timeout is in force
        with wd.phase("upload checkpoint", 5.0):
            clock.t += 0.6  # the upload: no steps, way past the tight timeout
            time.sleep(0.1)
            assert exits == []
        clock.t += 0.5  # phase over, so the tight timeout is back
        _settle(exits)
    assert exits == [70]
    assert "watchdog 0.2s" in stalls[0]


def test_phase_bounds_the_span_rather_than_exempting_it(clock: _FakeClock):
    stalls: list[str] = []
    exits: list[int] = []
    wd = Watchdog(0.2, stalls.append, _exit=exits.append)
    with wd:
        wd.poke(1, 10)
        with wd.phase("upload checkpoint", 1.0):
            clock.t += 0.5  # past the tight timeout the loop was held to…
            time.sleep(0.1)
            assert exits == []  # …and the span's own budget is what applies instead
            clock.t += 0.7  # an upload that hangs is still caught, at that budget
            _settle(exits)
    assert exits == [70]
    (diagnosis,) = stalls
    assert "blocking phase 'upload checkpoint' 1s" in diagnosis
    assert "at step 1/10" in diagnosis  # where the task was when the span began


def test_phase_only_ever_widens_the_threshold():
    wd = Watchdog(10.0, lambda _: None, grace_s=30.0)
    assert wd._threshold() == 30.0  # no poke yet: the startup grace governs
    with wd.phase("small put", 1.0):
        assert wd._threshold() == 30.0  # a narrower budget can't tighten the grace
        with wd.phase("big put", 60.0):
            assert wd._threshold() == 60.0
        assert wd._threshold() == 30.0  # the inner phase popped its own entry
    wd.poke(1, 10)
    assert wd._threshold() == 10.0


def test_stall_during_grace_names_the_grace(clock: _FakeClock):
    stalls: list[str] = []
    exits: list[int] = []
    wd = Watchdog(5.0, stalls.append, grace_s=0.2, _exit=exits.append)
    with wd:
        clock.t += 0.5  # never emits: the grace, not the timeout, is what expires
        _settle(exits)
    assert exits == [70]
    (diagnosis,) = stalls
    assert "startup grace 0.2s" in diagnosis
    assert "before any progress emission" in diagnosis


# ---------------------------------------------------------------------------
# Record surfacing: progress_at / steps_per_min / staleness views
# ---------------------------------------------------------------------------


def _msg(step: int, total: int = 10) -> ProgressMessage:
    return ProgressMessage(run_id="r", job_id="j", step=step, total=total)


def test_sink_splits_heartbeat_from_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = MemoStore(tmp_path / "sink")
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])

    sink = _MemoSink(store, "k")
    sink.put(_msg(1))  # anchors the rate window; first emission is an advance
    now[0] = 1030.0
    sink.put(_msg(4))
    assert store.record("k").get("steps_per_min") is None  # window (60s) not yet elapsed
    now[0] = 1070.0
    sink.put(_msg(8))
    rec = store.record("k")
    assert rec["progress_at"] == 1070.0
    assert rec["steps_per_min"] == pytest.approx(6.0)  # (8-1) steps over 70s
    now[0] = 1080.0
    sink.put(_msg(8))  # same step: heartbeat advances, progress doesn't
    rec = store.record("k")
    assert (rec["heartbeat_at"], rec["progress_at"]) == (1080.0, 1070.0)


def test_stale_progress_is_the_wedge_signature():
    base = {"state": RunState.RUNNING, "env": {"host": "x"}, "started_at": 900.0}
    fresh_hb = {**base, "heartbeat_at": 999.0, "progress_at": 998.0}
    wedged = {**base, "heartbeat_at": 999.0, "progress_at": 500.0}  # beating, not moving
    assert stale_progress(fresh_hb, now=1000.0) is False
    assert stale_progress(wedged, now=1000.0) is True
    assert progress_age(wedged, now=1000.0) == pytest.approx(500.0)
    # The worker's own watchdog threshold wins over the generic one when stamped.
    assert stale_progress({**wedged, "watchdog_s": 600.0}, now=1000.0) is False
    # No emission yet: age anchors on started_at; queued (no env) has no referent.
    assert progress_age(base | {"heartbeat_at": 901.0}, now=1000.0) == pytest.approx(100.0)
    assert progress_age({"state": RunState.RUNNING, "heartbeat_at": 901.0}, now=1000.0) is None


def test_store_transfers_declare_a_phase_sized_from_the_payload(tmp_path: Path):
    """A step that moves bytes gets its watchdog allowance without asking: ``put`` and ``get`` size a budget from the payload, so an experiment checkpointing after its last step needs no code of its own."""
    declared: list[tuple[str, float]] = []

    @contextmanager
    def spy(label: str, timeout_s: float) -> Iterator[None]:
        declared.append((label, timeout_s))
        yield

    store = LocalStore(tmp_path / "store")
    payload = tmp_path / "model"
    payload.mkdir()
    (payload / "weights.bin").write_bytes(b"x" * (4 << 20))  # 4 MiB

    with progress_context("r", "j", queue=None, emission_interval=0.1, on_phase=spy):
        art = store.put(payload, name="model")
        store.get(art, tmp_path / "back")

    # 120s overhead + 4 MiB at the 512 KiB/s floor = 8s. One phase per top-level
    # call: the per-child recursion runs on pool threads, which carry no job context.
    assert declared == [("put model", 128.0), ("get model", 128.0)]


def test_a_body_of_nothing_but_transfers_never_looks_wedged(tmp_path: Path):
    """The ex-2.1.11 shape, end to end: `publish_ablations` downloaded 27 artifacts and uploaded one, emitting no step progress for its entire run, and was badged stale three times while healthy. Nothing here calls `emit_progress` — the transfers' own spans are the only liveness the record ever gets."""
    store = LocalStore(tmp_path / "store")
    payload = tmp_path / "arr"
    payload.mkdir()
    (payload / "a.bin").write_bytes(b"x" * (1 << 20))

    started = time.time() - 700.0  # 12 minutes in, well past both thresholds
    rec: dict[str, Any] = {
        "state": RunState.RUNNING,
        "env": {"host": "worker.test"},
        "started_at": started,
        "heartbeat_at": started,
        "watchdog_s": 120.0,
    }

    def record(**fields: Any) -> bool:
        rec.update(fields)
        return True

    badges: list[tuple[bool, bool]] = []
    with progress_context("r", "j", queue=None, emission_interval=0.1, on_phase=_phase_hook(None, record)):
        for i in range(3):
            art = store.put(payload, name=f"arr{i}")
            badges.append((stale_heartbeat(rec), stale_progress(rec)))  # sampled in a gap
            store.get(art, tmp_path / f"back{i}")
            badges.append((stale_heartbeat(rec), stale_progress(rec)))
    badges.append((stale_heartbeat(rec), stale_progress(rec)))  # after the last span closes

    assert badges == [(False, False)] * len(badges)


@pytest.mark.parametrize(
    ("badge", "rec"),
    [
        # Step frozen for 300s under a 120s watchdog reads as a wedge — unless the
        # task said it was uploading, in which case it's healthy until that budget
        # runs out.
        pytest.param(
            stale_progress,
            {
                "state": RunState.RUNNING,
                "env": {"host": "x"},
                "started_at": 400.0,
                "heartbeat_at": 999.0,
                "progress_at": 700.0,
                "watchdog_s": 120.0,
            },
            id="stale_progress",
        ),
        # Heartbeat 700s old under the 300s threshold: dead, unless a span explains it.
        pytest.param(
            stale_heartbeat,
            {
                "state": RunState.RUNNING,
                "env": {"host": "x"},
                "started_at": 300.0,
                "heartbeat_at": 300.0,
                "watchdog_s": 120.0,
            },
            id="stale_heartbeat",
        ),
    ],
)
def test_a_declared_blocking_phase_pauses_both_wedge_badges(badge, rec):
    """A task whose whole body is store transfers emits no progress at all, so its heartbeat sits at ``started_at`` while it is perfectly healthy — the shape a publish step fanning in has (ex-2.1.11: 27 downloads and one upload, badged dead three times). The declared span is what separates that from a worker that really died, and it has to reach both badges or one still says wedged."""
    assert badge(rec, now=1000.0) is True
    assert badge(rec | {"phase_until": 1200.0}, now=1000.0) is False
    # The span bounds the silence rather than exempting it: past its own budget, a
    # worker that died mid-upload is flagged again.
    assert badge(rec | {"phase_until": 900.0}, now=1000.0) is True


def test_stale_progress_counts_a_recently_closed_phase_as_movement():
    """Between two consecutive transfers the record carries no deadline at all, and the step has not moved since ``started_at`` — so the gaps, not the spans, are where a transfer-only task gets called wedged. Crossing a boundary is the movement that says otherwise."""
    rec = {
        "state": RunState.RUNNING,
        "env": {"host": "x"},
        "started_at": 300.0,
        "heartbeat_at": 999.0,
        "progress_at": 700.0,
        "watchdog_s": 120.0,
    }
    assert stale_progress(rec, now=1000.0) is True  # step frozen, nothing to explain it
    assert stale_progress(rec | {"phase_at": 990.0}, now=1000.0) is False  # closed a span 10s ago
    # It buys one threshold, exactly as a step emission would — a task that closes a
    # span and *then* wedges is still caught.
    assert stale_progress(rec | {"phase_at": 600.0}, now=1000.0) is True


def test_phase_transitions_stamp_liveness_and_hand_nested_spans_back(monkeypatch: pytest.MonkeyPatch):
    """The span itself is covered by its deadline; what the stamps buy is the moments either side of it, where a task made only of transfers holds no label and no budget.

    And task code wrapping a `put` — which declares one of its own — leaves two spans open. The record holds one label, so the inner exit has to restore the outer's, not clear it: otherwise the badge calls the rest of the outer span a wedge.
    """
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    stamps: list[dict[str, Any]] = []

    def record(**fields: Any) -> bool:  # matches execute_task's own record()
        stamps.append(fields)
        return True

    phase = _phase_hook(None, record)
    with phase("archive epoch", 600.0):
        assert (stamps[-1]["phase"], stamps[-1]["phase_until"]) == ("archive epoch", 1600.0)
        with phase("put model", 120.0):
            pass
        assert (stamps[-1]["phase"], stamps[-1]["phase_until"]) == ("archive epoch", 1600.0)  # back to the outer span
    assert (stamps[-1]["phase"], stamps[-1]["phase_until"]) == (None, None)  # only the last exit clears it
    assert all(s["heartbeat_at"] == 1000.0 and s["phase_at"] == 1000.0 for s in stamps)  # every transition


def test_status_json_surfaces_the_declared_phase():
    """A frozen step reported alongside ``stale_progress: false`` reads as a contradiction unless the span that explains it travels with it."""
    from mini.__main__ import _task_json

    out = _task_json(
        {
            "key": "k",
            "fn": "train",
            "state": RunState.RUNNING,
            "env": {"host": "x"},
            "started_at": 400.0,
            "heartbeat_at": 999.0,
            "progress_at": 700.0,  # long stale in wall-clock terms
            "watchdog_s": 120.0,
            "step": 3300,
            "total": 3300,
            "phase": "put model",
            "phase_until": time.time() + 600.0,
        }
    )
    assert (out["phase"], out["step"], out["stale_progress"]) == ("put model", 3300, False)


def test_stale_progress_honors_startup_grace():
    # In setup (no progress_at yet): the grace governs, mirroring the watchdog.
    setup = {"state": RunState.RUNNING, "env": {"host": "x"}, "started_at": 400.0, "heartbeat_at": 999.0}
    thresholds = {"watchdog_s": 120.0, "watchdog_grace_s": 900.0}
    assert stale_progress(setup | thresholds, now=1000.0) is False  # 600s into a 900s grace
    assert stale_progress(setup | thresholds, now=1400.0) is True  # grace itself blown
    assert stale_progress(setup | {"watchdog_s": 120.0}, now=1000.0) is True  # no grace: tight from the start
    # Once emitting, the tight threshold applies regardless of the grace.
    running = setup | thresholds | {"progress_at": 700.0}
    assert stale_progress(running, now=1000.0) is True  # 300s > watchdog 120s, grace irrelevant now


# ---------------------------------------------------------------------------
# End to end: a wedged task aborts fast; healthy siblings are untouched
# ---------------------------------------------------------------------------


def test_wedged_worker_settles_failed_with_stack_dump(tmp_path: Path):
    # Local def so cloudpickle serializes it by value for the detached worker.
    def wedge_or_work(delay: float):
        from mini import emit_progress

        emit_progress(1, 100)
        time.sleep(delay)  # delay≥60 stands in for a hung device call
        emit_progress(100, 100)
        return delay

    def main(ctx):
        return ctx.map(wedge_or_work, [0.0, 60.0])

    exp = Experiment(name="wedge", main=main)
    app = LocalApparatus("wedge", max_workers=2, data_dir=tmp_path / "wedge").w(watchdog=0.5)
    store = app.memo_store()
    done, _ = tick(exp, app)
    assert not done  # both cells launched, in flight

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        by_state = {RunState(r["state"]): r for r in store.records() if r.get("state")}
        if RunState.FAILED in by_state and RunState.DONE in by_state:
            break
        time.sleep(0.2)
    else:
        raise AssertionError(f"expected one DONE + one FAILED, got: {store.records()}")

    wedged = by_state[RunState.FAILED]
    assert wedged["exc_type"] == "mini._watchdog.WatchdogStall"
    assert "no step progress" in wedged["error"]
    assert wedged["watchdog_s"] == 0.5
    diagnosis = store.error(wedged["key"])
    assert "wedge_or_work" in diagnosis  # the stack dump names the wedged frame
    # The healthy sibling finished normally under the same watchdog.
    assert by_state[RunState.DONE].get("exc_type") is None


def test_grace_lets_slow_setup_finish_under_a_tight_watchdog(tmp_path: Path):
    def slow_setup(x: int):
        from mini import emit_progress

        time.sleep(0.5)  # "tokenizing": longer than the watchdog, inside the grace
        for step in range(1, 4):
            emit_progress(step, 3)
        return x

    def main(ctx):
        return ctx.run(slow_setup, 1)

    exp = Experiment(name="grace", main=main)
    app = LocalApparatus("grace", data_dir=tmp_path / "grace").w(watchdog=0.25, watchdog_grace=30)
    store = app.memo_store()
    done, _ = tick(exp, app)
    assert not done

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        rec = store.records()[0]
        if rec.get("state") in (RunState.DONE, RunState.FAILED):
            break
        time.sleep(0.2)
    assert rec["state"] == RunState.DONE  # the 0.25s watchdog didn't kill the 0.5s setup
    assert (rec["watchdog_s"], rec["watchdog_grace_s"]) == (0.25, 30)


def test_blocking_phase_lets_a_post_loop_upload_finish(tmp_path: Path):
    """The ex-2.1.7 failure: training reached its last step, then the checkpoint push to the artifact store took longer than the gap the watchdog allows between steps — so a *finished* run was aborted and had to be re-trained. A declared phase covers the tail, and the run settles DONE."""

    def train_then_upload(x: int):
        from mini import blocking_phase, emit_progress

        for step in range(1, 4):
            emit_progress(step, 3)
        with blocking_phase("upload checkpoint", timeout_s=30.0):
            time.sleep(0.5)  # the push: no steps, and well past the 0.25s watchdog
        return x

    def main(ctx):
        return ctx.run(train_then_upload, 1)

    exp = Experiment(name="upload", main=main)
    app = LocalApparatus("upload", data_dir=tmp_path / "upload").w(watchdog=0.25)
    store = app.memo_store()
    done, _ = tick(exp, app)
    assert not done

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        rec = store.records()[0]
        if rec.get("state") in (RunState.DONE, RunState.FAILED):
            break
        time.sleep(0.2)
    assert rec["state"] == RunState.DONE, rec.get("error")
    assert rec.get("phase") is None  # the span closed, so the record isn't left claiming one


def test_backend_rerun_of_settled_attempt_is_a_noop(tmp_path: Path):
    """After a watchdog abort, Modal sees a container crash and re-schedules the input (regardless of retries=0). The re-run carries the same gen, so it must not flip the settled FAILED back to RUNNING and wedge again — it runs nothing and returns, ending the reschedule loop."""
    from mini._taskworker import execute_task
    from mini.memo import task_key_parts

    def boom():  # what the re-run would execute if the guard failed
        (tmp_path / "ran").touch()

    store = MemoStore(tmp_path / "rerun")
    key, parts = task_key_parts(boom, ())
    gen = store.mark_running(boom, key, parts, expect_gen=None)
    store.update(key, state=RunState.FAILED, error="WatchdogStall: …")  # the abort settled it

    execute_task(store, key, boom, (), [], gen=gen)
    assert store.record(key)["state"] == RunState.FAILED  # not resurrected to RUNNING
    assert not (tmp_path / "ran").exists()


def test_cancel_by_key_leaves_siblings_running(tmp_path: Path):
    def linger(x):
        time.sleep(30.0)
        return x

    def main(ctx):
        return ctx.map(linger, [1, 2])

    exp = Experiment(name="onecancel", main=main)
    app = LocalApparatus("onecancel", max_workers=2, data_dir=tmp_path / "onecancel")
    store = app.memo_store()
    done, _ = tick(exp, app)
    assert not done  # both cells launched, in flight
    k1, k2 = sorted(r["key"] for r in store.records())

    assert app.cancel(store, keys=[k1]) == [k1]
    states = {r["key"]: RunState(r["state"]) for r in store.records()}
    assert states == {k1: RunState.CANCELLED, k2: RunState.RUNNING}
    assert app.cancel(store) == [k2]  # cleanup: the unbounded form still sweeps the rest

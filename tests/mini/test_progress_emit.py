"""Progress emission must never charge the task for the sink's latency.

The failure this guards against was measured, not imagined: with the emission running inline, containers far from the control plane ran identical training cells 15–30× slower than their siblings, in order of distance, because every step paid a network round-trip. So the contract is: emitting is cheap for the caller whatever the sink costs, the *last* update still lands before the job ends, and a slow sink is never mistaken for a wedged worker.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from mini import _taskworker
from mini._debounce import BackgroundEmitter
from mini._taskworker import _MemoSink
from mini.memo import MemoStore
from mini.progress import ProgressMessage, emit_metrics, emit_progress, expect_metrics, progress_context


class _Sink:
    """The write half of ``QueueLike`` — progress never reads back."""

    def get(self, /, block: bool = True, timeout: float | None = None):
        raise NotImplementedError

    def empty(self) -> bool:
        return True


class SlowSink(_Sink):
    """A sink that takes *delay* seconds per delivery, recording what it received."""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.seen: list = []
        self._lock = threading.Lock()

    def put(self, item, /, block: bool = True, timeout: float | None = None) -> None:
        time.sleep(self.delay)
        with self._lock:
            self.seen.append(item)


def test_emitting_does_not_wait_for_the_sink():
    """100 emissions into a 20 ms sink must cost the caller ~nothing — inline that would be two seconds of training time — and only the survivors get delivered."""
    sink = SlowSink(delay=0.02)
    emitter = BackgroundEmitter(sink.put, interval=0.0)
    t0 = time.monotonic()
    for i in range(100):
        emitter(i)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.2, f"caller blocked on the sink ({elapsed:.2f}s for 100 emissions)"
    assert emitter.flush(timeout=5.0)
    assert len(sink.seen) < 100, "latest-wins should have dropped superseded updates"
    assert sink.seen[-1] == 99, "the final update must survive"
    emitter.close()


def test_flush_waits_for_the_last_update():
    """A task's final progress should be on the record before it settles."""
    sink = SlowSink(delay=0.1)
    emitter = BackgroundEmitter(sink.put, interval=0.0)
    emitter("last")
    assert emitter.flush(timeout=5.0)
    assert sink.seen == ["last"]
    emitter.close()


def test_sink_failures_do_not_reach_the_caller():
    """Progress is diagnostic: a control plane having a bad minute must not take the task down (and the caller is off the thread by then anyway)."""
    calls: list[int] = []

    def boom(i: int) -> None:
        calls.append(i)
        raise RuntimeError("control plane unavailable")

    emitter = BackgroundEmitter(boom, interval=0.0)
    emitter(1)
    emitter.flush(timeout=5.0)
    emitter(2)
    emitter.flush(timeout=5.0)
    emitter.close()
    assert calls == [1, 2], "a failed delivery must not stop later ones"


def test_watchdog_sees_progress_through_a_stalled_sink():
    """The watchdog measures the *task*, not the control plane: with the sink wedged, a task that keeps stepping must keep poking it — otherwise the background emitter would turn a network problem into a killed training run."""
    poked: list[tuple[int, int]] = []
    blocked = threading.Event()

    class WedgedSink(_Sink):
        def put(self, item, /, block: bool = True, timeout: float | None = None) -> None:
            blocked.wait(timeout=5.0)  # never delivers during the test

    with progress_context(
        "run", "job", queue=WedgedSink(), emission_interval=0.0, on_progress=lambda s, t: poked.append((s, t))
    ):
        for step in range(1, 6):
            emit_progress(step, 5)
        assert poked == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
        blocked.set()  # unwedge, so leaving the context doesn't wait out the flush


def test_metrics_ride_along_with_progress():
    """``emit_metrics`` merges into the job's metrics and travels on the next update, so a monitor reads numbers rather than parsing a message string."""
    sink = SlowSink(delay=0.0)
    with progress_context("run", "job", queue=sink, emission_interval=0.0):
        emit_metrics(loss=0.5)
        emit_progress(1, 10)
        emit_metrics(loss=0.25)
        emit_progress(2, 10)
    assert sink.seen[-1].metrics == {"loss": 0.25}
    assert sink.seen[-1].step == 2


# ---------------------------------------------------------------------------
# Which way a metric is meant to move — declared by the job, judged at the worker
# ---------------------------------------------------------------------------


def test_declared_directions_travel_with_the_numbers():
    """A name can't say whether a number should climb, so the job says it, and the declaration rides on every update from then on."""
    sink = SlowSink(delay=0.0)
    with progress_context("run", "job", queue=sink, emission_interval=0.0):
        expect_metrics(accuracy="up", loss="down")
        emit_metrics(accuracy=0.4, loss=2.0)
        emit_progress(1, 10)
    assert sink.seen[-1].goals == {"accuracy": "up", "loss": "down"}

    with pytest.raises(ValueError, match="'up' or 'down'"):
        expect_metrics(accuracy="higher")


def _windows(
    store: MemoStore, key: str, monkeypatch, series: list[tuple[float, float]], metric: str = "loss", **goals: str
) -> dict:
    """Feed ``(time, value)`` samples through a sink and return the final record.

    The rate window is 60 s, so a sample ≥60 s after the window opened closes it. The sample floor is dropped to one for these, so a handful of hand-picked values can make a point about *which way* a window read without twenty samples of padding around each. The floor has its own test below.
    """
    monkeypatch.setattr(_taskworker, "_MIN_WINDOW_SAMPLES", 1)
    now = [series[0][0]]
    monkeypatch.setattr(time, "time", lambda: now[0])
    sink = _MemoSink(store, key)
    for step, (t, value) in enumerate(series, start=1):
        now[0] = t
        sink.put(ProgressMessage(run_id="r", job_id="j", step=step, total=999, metrics={metric: value}, goals=goals))
    return store.record(key)


def test_a_window_is_judged_by_its_mean_not_its_last_sample(tmp_path: Path, monkeypatch):
    """A per-step loss is noisy, and comparing the two samples that happen to land on the window boundaries is a coin flip. Here the loss is flat — every window means exactly 2.0 — but the closing samples climb 1, 2, 3, 4, so endpoint comparison would have called three consecutive rises and flagged a run doing nothing wrong."""
    rec = _windows(
        MemoStore(tmp_path / "mean"),
        "k",
        monkeypatch,
        [
            (1000.0, 3.0), (1030.0, 2.0), (1060.0, 1.0),  # window 1 closes: mean 2.0, last 1.0
            (1090.0, 2.0), (1120.0, 2.0),                 # window 2 closes: mean 2.0, last 2.0
            (1150.0, 1.0), (1180.0, 3.0),                 # window 3 closes: mean 2.0, last 3.0
            (1210.0, 0.0), (1240.0, 4.0),                 # window 4 closes: mean 2.0, last 4.0
        ],
        loss="down",
    )  # fmt: skip
    assert rec["metrics_delta"] == {"loss": 0.0}
    assert rec["metrics_wrong_way"] == {}, "a flat loss must not read as a rising one"


@pytest.mark.parametrize(
    ("metric", "goal", "wrong_way"),
    [
        pytest.param("loss", "down", {"loss": 3}, id="climb-against-the-goal"),
        pytest.param("accuracy", "up", {}, id="climb-with-the-goal"),
    ],
)
def test_a_metric_going_the_wrong_way_accumulates_windows(tmp_path: Path, monkeypatch, metric, goal, wrong_way):
    """Steadily climbing where it should fall, window over window — the count is what ``status`` thresholds on, so one bad window stays quiet and a trend doesn't. The identical series declared the other way is a metric doing its job, so only the declaration separates the two.

    Five samples a minute apart close four windows, and the first close only *sets* the anchor there's nothing yet to compare against — so the count is three. Worth knowing when reading a young run: the flag needs one window more than its threshold before it can fire."""
    rec = _windows(
        MemoStore(tmp_path / metric),
        "k",
        monkeypatch,
        [(1000.0 + 60 * i, 1.0 + i) for i in range(5)],
        metric=metric,
        **{metric: goal},
    )
    assert rec["metrics_wrong_way"] == wrong_way
    assert rec["metric_goals"] == {metric: goal}
    assert rec["metrics_delta"] == {metric: 1.0}, "movement is still reported either way"


def test_an_undeclared_metric_is_measured_but_not_judged(tmp_path: Path, monkeypatch):
    """A domain-specific score gives away nothing about which way it ought to go, so an undeclared one records its movement and is never flagged — silence beats a wrong guess. The few names that can only mean one thing are the exception."""
    series = [(1000.0 + 60 * i, 1.0 + i) for i in range(5)]

    quiet = _windows(MemoStore(tmp_path / "mute"), "k", monkeypatch, series, metric="rho")
    assert quiet["metrics_delta"] == {"rho": 1.0} and quiet["metrics_wrong_way"] == {}
    assert quiet["metric_goals"] == {}

    guessed = _windows(MemoStore(tmp_path / "guess"), "k", monkeypatch, series, metric="val_loss")
    assert guessed["metric_goals"] == {"val_loss": "down"}
    assert guessed["metrics_wrong_way"] == {"val_loss": 3}, "a climbing loss needs no declaration"

    # …and a name that merely *contains* one of them is not guessed at: a
    # mixed-precision loss scale climbs by design.
    scale = _windows(MemoStore(tmp_path / "scale"), "k", monkeypatch, series, metric="loss_scale")
    assert scale["metric_goals"] == {} and scale["metrics_wrong_way"] == {}


def _feed(store: MemoStore, key: str, monkeypatch, per_minute: int, minutes: int) -> dict:
    """Run a job that emits *per_minute* samples of a climbing loss, for *minutes*."""
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    sink = _MemoSink(store, key)
    for step in range(per_minute * minutes):
        now[0] = 1000.0 + step * 60.0 / per_minute
        sink.put(ProgressMessage("r", "j", step + 1, 999, metrics={"loss": 1.0 + step}, goals={"loss": "down"}))
    return store.record(key)


def test_a_metric_window_waits_for_enough_samples_to_mean_something(tmp_path: Path, monkeypatch):
    """A window's verdict is a comparison of means, so it's worth what the sample count makes it worth — and that has nothing to do with elapsed time. A job emitting twice a minute would otherwise have two-sample means judged as confidently as a training loop's three hundred.

    So the window closes on the rate interval *or* the sample floor, whichever comes later. Both jobs below climb identically for ten minutes; the fast one fills a window a minute and flags it, the slow one has yet to reach a verdict — and gets there later, on its own terms, rather than never."""
    fast = _feed(MemoStore(tmp_path / "fast"), "k", monkeypatch, per_minute=60, minutes=10)
    assert fast["metrics_wrong_way"] == {"loss": 8}, "a window a minute, minus the one that set the anchor"

    slow = _feed(MemoStore(tmp_path / "slow"), "k", monkeypatch, per_minute=2, minutes=10)
    assert "metrics_wrong_way" not in slow, "no window has closed, so there is no verdict to publish yet"
    assert slow["steps_per_min"] == 2.0, "the throughput rate still reports on its own 60 s cadence"

    # The window stretched in time rather than going quiet: same emission rate,
    # long enough to fill several windows, and the climb is flagged.
    patient = _feed(MemoStore(tmp_path / "patient"), "k", monkeypatch, per_minute=2, minutes=60)
    assert patient["metrics_wrong_way"]["loss"] >= 3, "past the count `status` flags on"

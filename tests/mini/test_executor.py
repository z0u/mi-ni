"""Tests for the executor module."""

import time

from mini.executor import LocalExecutor, ProgressDisplay, get_progress


def test_local_executor_basic():
    """LocalExecutor.map runs the function and returns results in order."""
    executor = LocalExecutor('test', max_workers=2)
    results = list(executor.map(lambda x: x * 2, [1, 2, 3]))
    assert results == [2, 4, 6]


def test_local_executor_empty():
    """LocalExecutor.map handles empty input."""
    executor = LocalExecutor('test', max_workers=1)
    results = list(executor.map(lambda x: x, []))
    assert results == []


def test_local_executor_single_worker():
    """LocalExecutor with max_workers=1 runs sequentially."""
    executor = LocalExecutor('test', max_workers=1)
    call_order = []

    def track(x):
        call_order.append(x)
        return x

    results = list(executor.map(track, [1, 2, 3]))
    assert results == [1, 2, 3]
    assert call_order == [1, 2, 3]


def test_local_executor_progress_context():
    """Mapped functions can access progress via get_progress()."""
    executor = LocalExecutor('test', max_workers=1)

    def fn_with_progress(x):
        progress = get_progress()
        assert progress is not None
        progress.set_total(10)
        for i in range(10):
            progress.update(1, message=f'step {i}')
        return x

    results = list(executor.map(fn_with_progress, [1, 2]))
    assert results == [1, 2]


def test_progress_not_set_outside_executor():
    """get_progress() returns None when not inside an executor."""
    assert get_progress() is None


def test_local_executor_exception_propagates():
    """Exceptions in mapped functions propagate to the caller."""
    executor = LocalExecutor('test', max_workers=1)

    def fail(x):
        if x == 2:
            raise ValueError('bad value')
        return x

    results = []
    try:
        for r in executor.map(fail, [1, 2, 3]):
            results.append(r)
    except ValueError:
        pass
    assert results == [1]


def test_local_executor_concurrent():
    """LocalExecutor with multiple workers runs concurrently."""
    executor = LocalExecutor('test', max_workers=3)
    start = time.monotonic()

    def slow(x):
        time.sleep(0.1)
        return x

    results = list(executor.map(slow, [1, 2, 3]))
    elapsed = time.monotonic() - start
    assert results == [1, 2, 3]
    # With 3 workers, 3 jobs sleeping 0.1s should complete in ~0.1s, not ~0.3s
    assert elapsed < 0.25


def test_progress_display_lifecycle():
    """ProgressDisplay tracks job lifecycle."""
    display = ProgressDisplay(2)
    p0 = display.job_started(0)
    p0.set_total(5)
    p0.update(3, message='halfway')
    display.job_completed(0)

    display.job_started(1)
    display.job_failed(1, 'oops')
    display.finish()

    # Just verify it doesn't crash; output goes to stderr

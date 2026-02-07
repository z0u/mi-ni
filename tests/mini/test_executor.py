"""Tests for the executor module."""

import contextlib
import time

import pytest

from mini.executor import ProgressDisplay, get_progress
from mini.local_executor import LocalExecutor
from mini.modal_executor import ModalExecutor


# ---------------------------------------------------------------------------
# Mock Modal App — simulates Modal's behaviour so we can test ModalExecutor
# without network access.
# ---------------------------------------------------------------------------


class _MockModalFunction:
    """Simulates ``modal.Function`` produced by ``@app.function()``."""

    def __init__(self, fn):
        self._fn = fn

    def map(self, *input_iterators, kwargs=None, order_outputs=True, return_exceptions=False):
        kw = kwargs or {}
        for args in zip(*input_iterators, strict=False):
            yield self._fn(*args, **kw)


class MockModalApp:
    """Simulates ``modal.App`` for testing."""

    def __init__(self, name: str = 'test'):
        self.name = name

    def function(self, **decorator_kwargs):
        def decorator(fn):
            return _MockModalFunction(fn)

        return decorator

    def run(self):
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Fixtures — each test runs against both executors
# ---------------------------------------------------------------------------


def _make_local():
    return LocalExecutor('test', max_workers=1)


def _make_modal():
    return ModalExecutor(MockModalApp())


@pytest.fixture(params=['local', 'modal'], ids=['LocalExecutor', 'ModalExecutor'])
def executor(request):
    if request.param == 'local':
        return _make_local()
    return _make_modal()


# ---------------------------------------------------------------------------
# Parameter-passing tests — both executors must behave identically
# ---------------------------------------------------------------------------


def test_single_arg(executor):
    """map(fn, [a, b, c]) calls fn(a), fn(b), fn(c)."""
    results = list(executor.map(lambda x: x * 2, [1, 2, 3]))
    assert results == [2, 4, 6]


def test_two_args(executor):
    """map(fn, xs, ys) calls fn(x, y) for each pair."""
    results = list(executor.map(lambda x, y: f'{x}-{y}', [1, 2, 3], ['a', 'b', 'c']))
    assert results == ['1-a', '2-b', '3-c']


def test_single_arg_with_kwargs(executor):
    """map(fn, xs, kwargs={...}) forwards kwargs to every call."""

    def fn(x, scale=1):
        return x * scale

    results = list(executor.map(fn, [1, 2, 3], kwargs={'scale': 10}))
    assert results == [10, 20, 30]


def test_two_args_with_kwargs(executor):
    """map(fn, xs, ys, kwargs={...}) forwards both positional and keyword args."""

    def fn(x, y, sep=','):
        return f'{x}{sep}{y}'

    results = list(executor.map(fn, [1, 2], ['a', 'b'], kwargs={'sep': ':'}))
    assert results == ['1:a', '2:b']


def test_kwargs_only(executor):
    """map(fn, dummy_iter, kwargs={...}) works with functions that only use kwargs."""

    def fn(_, key='default'):
        return key

    results = list(executor.map(fn, range(3), kwargs={'key': 'hello'}))
    assert results == ['hello', 'hello', 'hello']


def test_no_kwargs(executor):
    """map(fn, xs) works without kwargs (kwargs defaults to None)."""

    def fn(x, y='default'):
        return f'{x}-{y}'

    results = list(executor.map(fn, [1, 2]))
    assert results == ['1-default', '2-default']


def test_empty(executor):
    """map with empty iterables returns no results."""
    results = list(executor.map(lambda x: x, []))
    assert results == []


def test_result_order_preserved(executor):
    """Results are returned in the same order as inputs, not completion order."""
    results = list(executor.map(lambda x: x**2, [3, 1, 4, 1, 5]))
    assert results == [9, 1, 16, 1, 25]


def test_complex_objects_as_args(executor):
    """map works with non-trivial argument types (dicts, dataclasses, etc.)."""

    def fn(params):
        return params['a'] + params['b']

    results = list(executor.map(fn, [{'a': 1, 'b': 2}, {'a': 10, 'b': 20}]))
    assert results == [3, 30]


# ---------------------------------------------------------------------------
# LocalExecutor-specific tests
# ---------------------------------------------------------------------------


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
    assert elapsed < 0.25


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


# ---------------------------------------------------------------------------
# ProgressDisplay tests
# ---------------------------------------------------------------------------


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

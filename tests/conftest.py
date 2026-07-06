# conftest.py — shared pytest fixtures and plugins

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_backend(monkeypatch):
    """A Modal-first shell must not steer the tests. `MINI_APP` would resolve
    CLI commands that omit `--app` onto the modal path (#47) — for a `run`
    test, real spawns — and the CLI's other-backend peek would touch the
    network on any empty read. Tests opt in to a backend explicitly (flags,
    markers, or mocks); a test of the hint itself re-patches `_peek`.
    """
    monkeypatch.delenv("MINI_APP", raising=False)
    monkeypatch.setattr("mini.__main__._peek", lambda name, backend: 0)

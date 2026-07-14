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


@pytest.fixture
def local_store(monkeypatch):
    """Force put/get/get_store onto a LocalStore, hermetically.

    A configured bucket resolves from two sources: the `MINI_STORE_BUCKET` /
    `MINI_PUBLISH_REPO` env vars *and* `[tool.mini] store-bucket` in the repo's
    `pyproject.toml` (`store_bucket`/`publish_repo`). Clearing only the env vars
    isn't enough — with the pyproject default plus an HF token in the ambient
    shell, `store_for` still returns an `HFStore` and diverts the CAS to the
    network. Neutralize the config-file fallback too, so store tests that assert
    against the local CAS don't depend on where they're run from.
    """
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    monkeypatch.setattr("mini.store._project_config", dict)

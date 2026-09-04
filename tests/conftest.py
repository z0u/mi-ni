# conftest.py — shared pytest fixtures and plugins

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

# JAX's persistent compilation cache, so a jitted train step compiles once per checkout rather than once per test process: the training tests build the same few HLO modules every run, and each XLA compile costs ~1.5s. Env vars rather than `jax.config`, so nothing imports JAX before a test asks for it (and so xdist workers share the setting). The default minimum compile time would skip exactly these small modules.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/mi-ni/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name: str) -> ModuleType:
    """Import `scripts/<name>.py` as a module. The scripts aren't a package, so tests load them by path, and the module is registered under its own name so `@dataclass` and `typing.get_type_hints` can resolve it."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _no_ambient_backend(monkeypatch):
    """A Modal-first shell must not steer the tests. `MINI_APP` would resolve CLI commands that omit `--app` onto the modal path (#47) — for a `run` test, real spawns — and the CLI's other-backend peek would touch the network on any empty read. Tests opt in to a backend explicitly (flags, markers, or mocks); a test of the hint itself re-patches `_peek`."""
    monkeypatch.delenv("MINI_APP", raising=False)
    monkeypatch.setattr("mini.__main__._peek", lambda name, backend: 0)


@pytest.fixture
def local_store(monkeypatch):
    """Force put/get/get_store onto a LocalStore, hermetically.

    A configured bucket resolves from two places: the `MINI_STORE_BUCKET` / `MINI_PUBLISH_REPO` env vars *and* `[tool.mini] store-bucket` in the repo's `pyproject.toml` or the gitignored `mini.local.toml` beside it (`store_bucket`/`publish_repo`). Clearing only the env vars isn't enough — with a config-file default plus an HF token in the ambient shell, `store_for` still returns an `HFStore` and diverts the CAS to the network. Switch the files off too (`MINI_NO_PROJECT_CONFIG`, which unlike a monkeypatch reaches spawned workers), so store tests that assert against the local CAS don't depend on where they're run from.
    """
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    monkeypatch.setenv("MINI_NO_PROJECT_CONFIG", "1")

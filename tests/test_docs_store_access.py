"""A report reads through `project_store()`; a bucket name under `docs/` is a bug.

Whichever bucket it is: the production name hardcodes what configuration already knows, and a dev name leaves a published report resolving against a sandbox that can be wiped. See the mi-ni skill's storage reference, "Which pair a run uses".
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def configured_names() -> set[str]:
    """Every `store-bucket`/`publish-repo` the project configures, base and profiles."""
    mini = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["mini"]
    tables = [mini, *mini.get("profiles", {}).values()]
    return {name for t in tables for key in ("store-bucket", "publish-repo") if (name := t.get(key))}


def offenders() -> dict[Path, set[str]]:
    """Notebooks under `docs/` that build a store or name a configured repo."""
    needles = configured_names() | {"HFStore("}
    found = {}
    for path in sorted(ROOT.glob("docs/**/*.py")):
        text = path.read_text()
        if hits := {n for n in needles if n in text}:
            found[path.relative_to(ROOT)] = hits
    return found


def test_no_report_names_a_bucket_or_builds_a_store():
    assert offenders() == {}

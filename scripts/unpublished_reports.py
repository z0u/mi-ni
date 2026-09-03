#!/usr/bin/env python
"""Report notebooks this branch changed without republishing — the forgotten-publish check.

``./go publish`` is the step of the loop that has to be remembered, and forgetting it fails quietly: the notebook merges, the site keeps serving the previous export's figures, and nothing says so. This is the tripwire. It compares what the branch *changed* against what it *repinned* in ``docs/publish.lock``, and names the gap. "Changed" covers a report's own directory, not just its notebook — see :func:`changed_reports`.

Both halves are cheap and local — a git diff and a JSON file — so the check needs no store access, no render, and no write credentials. That's the point: the publish itself stays where the data is (a session with a warm store), and CI only has to notice when it didn't happen.

The comparison is also self-correcting when ``origin/main`` is behind, which it often is in a fresh container: a stale base makes *more* reports look changed, but their pins moved in that same range, so they don't register as unpublished.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from mini.reports import (  # noqa: E402
    PUBLISH_LOCK,
    export_key,
    input_dir,
    is_manually_published,
    load_pins,
    report_notebooks,
)


def changed_reports(base: str, root: Path = ROOT) -> list[Path]:
    """The reports this branch changed since *base* — through their notebook, or through a file beside it.

    A report is dated by its inputs as much as by its own source: re-running an experiment writes new results and edits ``docs/<key>/experiment.py``, leaving ``report.py`` untouched, and the bundle then serves the previous run's figures. So the notebook's own directory counts as part of it (:func:`~mini.reports.input_dir`) — every changed file under it dates the report, except a *sibling report*, which is a second document rather than an input to the first.

    The diff is three-dot (``base...HEAD``), i.e. against the merge base, so commits that landed on the base branch meanwhile aren't mistaken for ours. Deletions need no filtering: the candidates come from the reports that exist *now*, so a deleted report simply isn't among them (it has no bundle to publish, and the next publish prunes its pin — ``export_reports.update_pins``), while a deleted input still dates the report it belonged to.

    Scoped to ``docs/`` by the same reasoning as :func:`~mini.reports.report_notebooks`: a report is a notebook *there*. Without the pathspec, anything in the repo carrying the text ``marimo.App(`` reads as a report — this module's own tests, for instance.
    """
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD", "--", "docs"],
        cwd=(root := root.resolve()),  # so these paths compare against input_dir's, which resolves
        capture_output=True,
        text=True,
    )
    if diff.returncode != 0:
        sys.exit(f"git diff against '{base}' failed — is that ref fetched?\n{diff.stderr.strip()}")
    touched = {root / line for line in diff.stdout.splitlines() if line}
    reports = report_notebooks(root / "docs")
    inputs = touched - set(reports)  # a sibling report is a second document, not an input to this one

    def dated(nb: Path) -> bool:
        return nb in touched or ((d := input_dir(nb)) is not None and any(d in p.parents for p in inputs))

    return sorted(nb for nb in reports if dated(nb) and not is_manually_published(nb))


def pins_at(base: str, root: Path = ROOT) -> dict[str, str]:
    """The pin manifest as of *base*, or ``{}`` if it didn't exist there yet."""
    show = subprocess.run(
        ["git", "show", f"{base}:{PUBLISH_LOCK.as_posix()}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return json.loads(show.stdout) if show.returncode == 0 else {}


def unpublished(base: str, root: Path = ROOT) -> list[Path]:
    """Changed reports whose pin hasn't moved since *base* — the ones to publish.

    A report that was never published reads as unpublished too: its key is absent from both manifests, so the pin is unchanged in the sense that matters. The exception is a project with no manifest at all: until the first ``./go publish`` writes one, there is nothing to hold reports to, so nothing is reported.
    """
    if not (root / PUBLISH_LOCK).exists():
        return []
    before, after = pins_at(base, root), load_pins(root)
    return [nb for nb in changed_reports(base, root) if after.get(key := export_key(nb)) == before.get(key)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="the ref to compare against, e.g. origin/main")
    args = ap.parse_args()

    for nb in unpublished(args.base):
        print(nb.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()

"""Tests for the forgotten-publish check — changed reports whose pin didn't move."""

import json
import subprocess
from pathlib import Path

import pytest
from mini.reports import MANUAL_PUBLISH_MARKER

from tests.conftest import load_script

unpub = load_script("unpublished_reports")

_APP = "import marimo\napp = marimo.App()\n"


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def commit(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def pin(repo: Path, key: str, revision: str) -> None:
    """Repin one report, as `./go publish` would."""
    lock = repo / "docs" / "publish.lock"
    pins = json.loads(lock.read_text()) | {key: revision}
    lock.write_text(json.dumps(dict(sorted(pins.items())), indent=1) + "\n")


def flagged(repo: Path, base: str = "main") -> set[str]:
    return {p.relative_to(repo).as_posix() for p in unpub.unpublished(base, root=repo)}


def changed(repo: Path, base: str = "main") -> set[str]:
    return {p.relative_to(repo).as_posix() for p in unpub.changed_reports(base, root=repo)}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Two published reports on `main`, with a `work` branch checked out."""
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "commit.gpgsign", "false")
    (docs := tmp_path / "docs" / "ex-1").mkdir(parents=True)
    (docs / "report.py").write_text(_APP)
    (docs / "experiment.py").write_text("def main(ctx): ...\n")
    (tmp_path / "docs" / "overview.py").write_text(_APP)
    (tmp_path / "docs" / "publish.lock").write_text(json.dumps({"ex-1": "aaa", "overview": "bbb"}, indent=1) + "\n")
    commit(tmp_path, "reports")
    git(tmp_path, "checkout", "-q", "-b", "work")
    return tmp_path


# --- which notebooks count as changed reports -------------------------------------


def test_nothing_changed(repo):
    assert changed(repo) == set()


def test_a_changed_input_beside_a_report_counts_as_changing_it(repo):
    """The re-run case: new results land via `experiment.py` while `report.py` sits still, and the bundle then serves the previous run's figures."""
    (repo / "docs" / "ex-1" / "experiment.py").write_text("def main(ctx): pass\n")
    commit(repo, "edit the experiment definition")
    assert changed(repo) == {"docs/ex-1/report.py"}


def test_an_input_in_a_nested_dir_counts_too(repo):
    (data := repo / "docs" / "ex-1" / "data").mkdir()
    (data / "dopesheet.csv").write_text("step,lr\n0,1e-3\n")
    commit(repo, "add a dopesheet")
    assert changed(repo) == {"docs/ex-1/report.py"}


def test_a_deleted_input_counts_too(repo):
    """A delete leaves no file to inspect, so the report is found from its directory rather than the path being attributed to it."""
    (repo / "docs" / "ex-1" / "experiment.py").unlink()
    commit(repo, "drop the experiment definition")
    assert changed(repo) == {"docs/ex-1/report.py"}


def test_shared_docs_files_belong_to_no_report(repo):
    """The docs root is site space. Reading it as one report's inputs would flag `overview.py` on every publish, since `publish.lock` lives there."""
    (repo / "docs" / "report.css").write_text(".marimo { color: red }\n")
    (repo / "docs" / "index.md").write_text("# Reports\n")
    pin(repo, "ex-1", "ccc")
    commit(repo, "restyle and repin")
    assert changed(repo) == set()


def test_a_sibling_report_is_a_document_not_an_input(repo):
    """Two reports in one directory: an input change dates both, but neither dates the other."""
    (repo / "docs" / "ex-1" / "aside.py").write_text(_APP)
    commit(repo, "a second report alongside the first")
    assert changed(repo) == {"docs/ex-1/aside.py"}

    (repo / "docs" / "ex-1" / "experiment.py").write_text("def main(ctx): pass\n")
    commit(repo, "edit the shared experiment definition")
    assert changed(repo) == {"docs/ex-1/aside.py", "docs/ex-1/report.py"}


def test_a_notebook_outside_docs_is_not_a_report(repo):
    """`marimo.App(` appears in plenty of files that aren't reports — this repo's own tests among them. Only `docs/` is the report tree."""
    (tests := repo / "tests").mkdir()
    (tests / "test_something.py").write_text(f'SAMPLE = """{_APP}"""\n')
    (repo / "notebook.py").write_text(_APP)
    commit(repo, "a test that quotes a notebook")
    assert changed(repo) == set()


def test_a_deleted_report_has_nothing_to_publish(repo):
    (repo / "docs" / "overview.py").unlink()
    commit(repo, "drop a report")
    assert changed(repo) == set()


def test_base_branch_commits_are_not_ours(repo):
    """Three-dot: the diff is against the merge base, so main moving on doesn't count."""
    git(repo, "checkout", "-q", "main")
    (repo / "docs" / "overview.py").write_text(_APP + "# landed on main meanwhile\n")
    commit(repo, "someone else's report")
    git(repo, "checkout", "-q", "work")
    assert changed(repo) == set()


# --- and of those, which were left unpublished --------------------------------------


def test_a_changed_report_without_a_new_pin_is_flagged(repo):
    (repo / "docs" / "ex-1" / "report.py").write_text(_APP + "# edited\n")
    commit(repo, "edit a report")
    assert changed(repo) == {"docs/ex-1/report.py"}
    assert flagged(repo) == {"docs/ex-1/report.py"}


def test_repinning_clears_it(repo):
    (repo / "docs" / "ex-1" / "report.py").write_text(_APP + "# edited\n")
    pin(repo, "ex-1", "ccc")
    commit(repo, "edit and publish a report")
    assert flagged(repo) == set()


def test_a_publish_under_a_profile_does_not_count(repo, monkeypatch):
    """A dev publish moves the dev manifest only; the report is still unpublished as far as the site knows."""
    (repo / "docs" / "ex-1" / "report.py").write_text(_APP + "# edited\n")
    (repo / ".mini").mkdir()
    (repo / ".mini" / "publish.dev.lock").write_text(json.dumps({"ex-1": "ccc"}))
    commit(repo, "edit a report, publish it to dev")
    monkeypatch.setenv("MINI_PROFILE", "dev")
    assert flagged(repo) == {"docs/ex-1/report.py"}


def test_only_the_unpinned_report_is_flagged(repo):
    (repo / "docs" / "ex-1" / "report.py").write_text(_APP + "# edited\n")
    (repo / "docs" / "overview.py").write_text(_APP + "# edited\n")
    pin(repo, "ex-1", "ccc")
    commit(repo, "publish one of the two")
    assert flagged(repo) == {"docs/overview.py"}


def test_a_re_run_experiment_is_flagged_until_republished(repo):
    (repo / "docs" / "ex-1" / "experiment.py").write_text("def main(ctx): pass\n")
    commit(repo, "re-run the experiment")
    assert flagged(repo) == {"docs/ex-1/report.py"}

    pin(repo, "ex-1", "ccc")
    commit(repo, "republish against the new results")
    assert flagged(repo) == set()


def test_a_brand_new_report_needs_its_first_publish(repo):
    (docs := repo / "docs" / "ex-2").mkdir()
    (docs / "report.py").write_text(_APP)
    commit(repo, "new report")
    assert flagged(repo) == {"docs/ex-2/report.py"}


def test_the_marker_opts_out_of_the_reminder(repo):
    (repo / "docs" / "ex-1" / "report.py").write_text(f"# {MANUAL_PUBLISH_MARKER}\n{_APP}# edited\n")
    commit(repo, "hand-published from here on")
    assert flagged(repo) == set()


def test_a_stale_base_does_not_produce_noise(repo):
    """A container's `origin/main` is often behind. That makes more reports look changed, but their pins moved in the same range, so they don't read as unpublished."""
    (repo / "docs" / "ex-1" / "report.py").write_text(_APP + "# edited\n")
    pin(repo, "ex-1", "ccc")
    commit(repo, "edit and publish")
    git(repo, "tag", "stale-base", "main")  # pretend main advanced past this point
    (repo / "docs" / "overview.py").write_text(_APP + "# edited later\n")
    commit(repo, "a later edit, unpublished")
    assert flagged(repo, "stale-base") == {"docs/overview.py"}


def test_unknown_base_ref_fails_loudly(repo):
    with pytest.raises(SystemExit, match="nope"):
        unpub.changed_reports("nope", root=repo)


def test_a_project_without_a_manifest_is_not_held_to_one(tmp_path: Path):
    """Until the first publish writes docs/publish.lock, a changed report is nothing to report."""
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "overview.py").write_text(_APP)
    commit(tmp_path, "report, never published")
    git(tmp_path, "checkout", "-q", "-b", "work")
    (tmp_path / "docs" / "overview.py").write_text(_APP + "# edited\n")
    commit(tmp_path, "edit")
    assert changed(tmp_path) == {"docs/overview.py"}
    assert flagged(tmp_path) == set()

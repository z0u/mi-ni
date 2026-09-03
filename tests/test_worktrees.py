"""Tests for the agent-worktree prune — what counts as finished, and what gets held back.

The interesting half is `judge`, so most of this runs against a real temporary repo with real worktrees rather than mocking git out. The decisions being checked are about git's actual behaviour (what a squash merge leaves behind, what `status --porcelain` counts), and a mock would only encode this file's guess at those.
"""

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from tests.conftest import load_script

wt = load_script("worktrees")


def run(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo on `main` with one commit, and identity set so commits work without a global config."""
    root = tmp_path / "repo"
    root.mkdir()
    run("init", "-q", "-b", "main", cwd=root)
    run("config", "user.email", "test@example.invalid", cwd=root)
    run("config", "user.name", "Test", cwd=root)
    (root / "file.txt").write_text("one\n")
    run("add", ".", cwd=root)
    run("commit", "-qm", "first", cwd=root)
    return root


def add_worktree(repo: Path, name: str, start: str = "main") -> Path:
    path = repo / wt.NEST / name
    run("worktree", "add", "-q", "-b", f"probe/{name}", str(path), start, cwd=repo)
    return path


def status_for(repo: Path, name: str) -> wt.Status:
    """The judgement for one named worktree, as the CLI would compute it."""
    trees = wt.parse(run("worktree", "list", "--porcelain", cwd=repo), root=repo)
    tree = next(t for t in trees if t.path.name == name)
    return wt.judge(tree, "main")


def test_a_worktree_level_with_main_is_finished(repo: Path):
    add_worktree(repo, "landed")
    assert status_for(repo, "landed").removable


def test_a_squash_merged_branch_reads_as_finished(repo: Path):
    """The case an ancestor test alone would miss, and the reason `judge` also compares trees.

    A squash merge leaves the branch holding commits that `main` never took, while the content of both ends up identical. Modelled here by committing a change and reverting it: novel commits, same tree.
    """
    path = add_worktree(repo, "squashed")
    (path / "extra.txt").write_text("x\n")
    run("add", ".", cwd=path)
    run("commit", "-qm", "add", cwd=path)
    run("rm", "-q", "extra.txt", cwd=path)
    run("commit", "-qm", "remove", cwd=path)

    assert not wt.git_ok("merge-base", "--is-ancestor", "HEAD", "main", cwd=path)  # the scenario is real
    status = status_for(repo, "squashed")
    assert status.landed and status.removable


def test_uncommitted_work_holds_a_worktree_back(repo: Path):
    path = add_worktree(repo, "midflight")
    (path / "file.txt").write_text("edited\n")

    status = status_for(repo, "midflight")
    assert status.dirty
    assert not status.removable
    assert status.why("main") == "uncommitted or untracked changes"


def test_untracked_scratch_files_count_as_uncommitted(repo: Path):
    """The shape a session that ended badly leaves: nothing staged, but work on disk all the same."""
    path = add_worktree(repo, "scratch")
    (path / "notes.md").write_text("half an idea\n")

    assert not status_for(repo, "scratch").removable


def test_ignored_files_do_not_hold_a_worktree_back(repo: Path):
    """The deliberate limit on the clean test, and the one that surprises.

    `__pycache__/` lands in a worktree the moment anything in it runs, so counting ignored files would make every worktree that was ever used unprunable. `.gitignore` is the repo naming what it is willing to lose, and this takes it at its word — which does mean a session's notes in a file matching `scratch*` go with the checkout.
    """
    (repo / ".gitignore").write_text("scratch*\n")
    run("add", ".gitignore", cwd=repo)
    run("commit", "-qm", "ignore scratch", cwd=repo)
    path = add_worktree(repo, "cached")
    (path / "scratch.txt").write_text("disposable\n")

    status = status_for(repo, "cached")
    assert not status.dirty
    assert status.removable


def test_commits_that_never_landed_hold_a_worktree_back(repo: Path):
    path = add_worktree(repo, "ahead")
    (path / "new.txt").write_text("real work\n")
    run("add", ".", cwd=path)
    run("commit", "-qm", "work nobody merged", cwd=path)

    status = status_for(repo, "ahead")
    assert not status.landed
    assert status.why("main") == "commits not in main"


def test_the_main_worktree_is_never_a_candidate(repo: Path):
    add_worktree(repo, "nested")
    trees = wt.parse(run("worktree", "list", "--porcelain", cwd=repo), root=repo)
    assert [t.path.name for t in trees] == ["nested"]


def test_a_worktree_outside_the_repo_is_left_alone(repo: Path, tmp_path: Path):
    """A sibling checkout is someone's deliberate workspace, and not what this cleans up."""
    sibling = tmp_path / "elsewhere"
    run("worktree", "add", "-q", "-b", "probe/sibling", str(sibling), "main", cwd=repo)

    trees = wt.parse(run("worktree", "list", "--porcelain", cwd=repo), root=repo)
    outside = next(t for t in trees if t.path.name == "elsewhere")
    assert not outside.nested
    status = wt.judge(outside, "main")
    assert status.landed and not status.dirty  # finished by every other measure...
    assert not status.removable  # ...and still left where it is
    assert status.why("main") == "outside the repo — left alone"


def test_a_locked_worktree_is_left_alone(repo: Path):
    """`git worktree lock` is an explicit "don't touch this", so it outranks looking finished."""
    add_worktree(repo, "held")
    run("worktree", "lock", str(repo / wt.NEST / "held"), cwd=repo)

    status = status_for(repo, "held")
    assert status.tree.locked
    assert not status.removable
    assert status.why("main") == "locked"


def test_a_locked_worktree_whose_directory_is_gone_is_dropped(repo: Path):
    """`git worktree prune` clears a stale entry, but not a locked one — and judging it would run `git status` in a directory that isn't there."""
    path = add_worktree(repo, "vanished")
    run("worktree", "lock", str(path), cwd=repo)
    shutil.rmtree(path)
    run("worktree", "prune", cwd=repo)  # leaves the locked entry behind

    listing = run("worktree", "list", "--porcelain", cwd=repo)
    assert "vanished" in listing  # git still knows about it...
    assert wt.parse(listing, root=repo) == []  # ...and we have nothing to say about it


def test_a_detached_worktree_parses_without_a_branch(repo: Path):
    head = run("rev-parse", "HEAD", cwd=repo)
    run("worktree", "add", "-q", "--detach", str(repo / wt.NEST / "loose"), head, cwd=repo)

    trees = wt.parse(run("worktree", "list", "--porcelain", cwd=repo), root=repo)
    assert next(t for t in trees if t.path.name == "loose").branch is None


def test_integration_ref_prefers_the_remote(repo: Path):
    """`origin/main` is the shared truth; a local `main` can sit behind it or ahead of it."""
    assert wt.integration_ref(cwd=repo) == "main"  # no remote configured in the fixture

    run("update-ref", "refs/remotes/origin/main", "HEAD", cwd=repo)
    assert wt.integration_ref(cwd=repo) == "origin/main"


def test_no_integration_ref_at_all_is_reported_not_guessed(tmp_path: Path):
    """With nothing to compare against, "finished" is unanswerable — so it isn't answered."""
    root = tmp_path / "bare"
    root.mkdir()
    run("init", "-q", "-b", "trunk", cwd=root)
    assert wt.integration_ref(cwd=root) is None


# The configs below are what keep the tree-walking tools out of a nested worktree, and each does
# it by a different mechanism. They are guards on config, not tests of the tools: running ruff,
# ty, vulture and pytest against a scratch worktree is how the mechanisms were established
# (2026-08-15), and it takes minutes, which is too slow to spend on every suite run. What these
# catch is the edit that quietly removes one of them — a failure that otherwise surfaces as a
# pile of unexplained errors in a checkout nobody remembers making.


def test_pytest_still_skips_dotted_directories():
    """Setting `norecursedirs` *replaces* pytest's defaults, and `.*` is the entry that matters.

    This is the one that already broke: listing `node_modules` and `.venv` dropped `.*` along with them, and pytest collected a second copy of `tests/` out of `.claude/worktrees/`, failing four `ImportPathMismatchError`s on every run.
    """
    config = tomllib.loads((wt.ROOT / "pyproject.toml").read_text())
    assert ".*" in config["tool"]["pytest"]["ini_options"]["norecursedirs"]


def test_nested_worktrees_are_gitignored():
    """What keeps `ruff` and `ty` out — both respect `.gitignore` by default, and neither names `.claude` in its own excludes.

    Verified by running each against a scratch worktree: clean as configured, and `ruff --no-respect-gitignore` picks up all 190 files in it, so the ignore file is doing the work rather than a default exclude list.
    """
    assert f"{wt.NEST}/" in (wt.ROOT / ".gitignore").read_text().splitlines()


def test_vulture_scans_named_roots_rather_than_the_whole_tree():
    """What keeps `vulture` out: it never gets offered the path, so its `exclude` list never has to mention one.

    `exclude` replaces rather than extends, same as pytest's, and lists only build detritus. Swapping these explicit roots for a bare `.` would walk the whole repo and inherit the problem.
    """
    paths = tomllib.loads((wt.ROOT / "pyproject.toml").read_text())["tool"]["vulture"]["paths"]
    assert paths and not any(p in {".", "./"} for p in paths)

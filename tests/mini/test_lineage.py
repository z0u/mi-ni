"""Run-lineage capture: git state, identity, driver env, and the merge/snapshot helpers.

The git tests build a throwaway repo under ``tmp_path`` so they exercise the real
``git`` plumbing (sha, dirty diff, remote sanitizing) without touching the project
repo. The identity/driver tests drive off an explicit ``env`` dict so they don't
depend on the ambient environment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mini import lineage


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Ada Lovelace")
    _git(root, "config", "user.email", "ada@example.com")
    (root / "a.txt").write_text("hello\n")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-qm", "first commit")
    return root


def test_git_lineage_clean_tree_pins_to_a_sha(repo: Path):
    lin = lineage.git_lineage(repo)
    assert lin is not None
    assert len(lin["sha"]) == 40
    assert lin["short_sha"] == lin["sha"][:12]
    assert lin["dirty"] is False
    assert "diff" not in lin  # nothing uncommitted to record
    assert lin["subject"] == "first commit"


def test_git_lineage_dirty_tree_carries_the_diff_and_untracked(repo: Path):
    (repo / "a.txt").write_text("hello\nworld\n")  # modify tracked
    (repo / "new.txt").write_text("secret-ish contents\n")  # untracked
    lin = lineage.git_lineage(repo)
    assert lin is not None
    assert lin["dirty"] is True
    assert "world" in lin["diff"]  # tracked change is captured verbatim
    assert lin["diff_truncated"] is False
    assert lin["untracked"] == ["new.txt"]  # name only — contents are NOT recorded
    assert "secret-ish contents" not in lin["diff"]


def test_git_lineage_tags_and_branch(repo: Path):
    _git(repo, "tag", "v1.2.3")
    _git(repo, "checkout", "-qb", "feature")
    lin = lineage.git_lineage(repo)
    assert lin is not None
    assert lin["branch"] == "feature"
    assert lin["tags"] == ["v1.2.3"]


def test_git_lineage_strips_remote_credentials(repo: Path):
    _git(repo, "remote", "add", "origin", "https://x-access-token:ghs_SECRET@github.com/z0u/mi-ni")
    lin = lineage.git_lineage(repo)
    assert lin is not None
    assert lin["remotes"]["origin"] == "https://github.com/z0u/mi-ni"
    assert "ghs_SECRET" not in str(lin)  # the token never lands in the record


def test_git_lineage_none_outside_a_repo(tmp_path: Path):
    assert lineage.git_lineage(tmp_path) is None


def test_operators_uses_repo_owner_handle_not_the_git_name(repo: Path):
    # An agent/CI committer identity is a bot, and a real name/email is PII — so the
    # operator is the non-PII repo owner from the remote, never user.name/user.email.
    _git(repo, "remote", "add", "origin", "https://github.com/z0u/mi-ni.git")
    assert lineage.operators(repo) == [{"handle": "z0u", "source": "git-remote"}]


def test_operators_empty_without_a_remote(repo: Path):
    assert lineage.operators(repo) == []  # no remote → no handle to attribute (never the git name)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/z0u/mi-ni.git",
        "git@github.com:z0u/mi-ni.git",
        "http://local_proxy@127.0.0.1:41729/git/z0u/mi-ni",
    ],
)
def test_operators_parses_owner_from_remote_shapes(repo: Path, url: str):
    _git(repo, "remote", "add", "origin", url)
    assert lineage.operators(repo) == [{"handle": "z0u", "source": "git-remote"}]


def test_sanitize_url_variants():
    assert (
        lineage._sanitize_url("http://local_proxy@127.0.0.1:41729/git/z0u/mi-ni")
        == "http://127.0.0.1:41729/git/z0u/mi-ni"
    )
    assert lineage._sanitize_url("https://user:pass@host/path") == "https://host/path"
    assert lineage._sanitize_url("git@github.com:z0u/mi-ni.git") == "git@github.com:z0u/mi-ni.git"  # scp-style: no ://


def test_agents_detects_claude_code_non_pii():
    env = {"CLAUDECODE": "1", "CLAUDE_CODE_VERSION": "2.1.42", "CLAUDE_CODE_ENTRYPOINT": "cli"}
    (agent,) = lineage.agents(env)
    assert agent == {"name": "claude-code", "version": "2.1.42", "entrypoint": "cli"}


def test_agents_empty_without_markers():
    assert lineage.agents({"PATH": "/usr/bin"}) == []


def test_driver_env_detects_runner_and_omits_secrets():
    env = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_WORKFLOW": "ci",
        "GITHUB_RUN_ID": "42",
        "GITHUB_TOKEN": "ghp_SHOULD_NOT_APPEAR",
    }
    drv = lineage.driver_env(env)
    assert drv["runner"] == {"kind": "github-actions", "workflow": "ci", "run_id": "42"}
    assert "ghp_SHOULD_NOT_APPEAR" not in str(drv)
    assert {"host", "platform", "python", "cpu_count"} <= drv.keys()


def test_merge_run_lineage_keeps_first_run_and_counts_wakes():
    first = {"captured_at": "T1", "captured_at_epoch": 1.0}
    merged1 = lineage.merge_run_lineage(None, dict(first))
    assert merged1["first_captured_at"] == "T1"
    assert merged1["wakes"] == 1

    second = {"captured_at": "T2", "captured_at_epoch": 2.0}
    merged2 = lineage.merge_run_lineage(merged1, dict(second))
    assert merged2["captured_at"] == "T2"  # latest snapshot wins
    assert merged2["first_captured_at"] == "T1"  # origin preserved
    assert merged2["wakes"] == 2


def test_merge_run_lineage_preserves_upstreams_across_a_wake():
    prev = lineage.merge_run_lineage(
        None, {"captured_at": "T1", "captured_at_epoch": 1.0, "upstreams": [{"experiment": "a"}]}
    )
    merged = lineage.merge_run_lineage(prev, {"captured_at": "T2", "captured_at_epoch": 2.0})
    assert merged["upstreams"] == [{"experiment": "a"}]  # a wake that didn't re-resolve deps keeps them


def test_upstream_snapshot_pulls_a_compact_trace():
    meta = {
        "lineage": {"git": {"sha": "abc123", "describe": "v1-dirty", "dirty": True}, "first_captured_at": "T0"},
        "modal_app_ids": ["ap-1", "ap-2"],
    }
    snap = lineage.upstream_snapshot("prep", meta)
    assert snap == {
        "experiment": "prep",
        "git_sha": "abc123",
        "git_describe": "v1-dirty",
        "git_dirty": True,
        "run_at": "T0",
        "modal_app_ids": ["ap-1", "ap-2"],
    }


def test_run_lineage_assembles_the_whole_snapshot(repo: Path):
    lin = lineage.run_lineage(repo)
    assert {"captured_at", "captured_at_epoch", "agents", "operators", "driver", "git"} <= lin.keys()
    assert lin["git"]["subject"] == "first commit"

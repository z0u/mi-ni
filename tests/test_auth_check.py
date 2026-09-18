"""Tests for the credential status probe (`./go auth --check`).

The parsing helpers are pure; the per-provider checks shell out through a single `_run`, so we drive them by swapping in a fake that returns canned tool output.
"""

import asyncio

import pytest

from tests.conftest import load_script

auth_check = load_script("auth_check")


def fake_scope(scope: str | None):
    """A stand-in for `_token_scope`, so a test of the detail line makes no network call."""

    async def _token_scope():
        return scope

    return _token_scope


def fake_run(code: int, out: str = "", err: str = ""):
    """A stand-in for `_run` that ignores the command and returns canned output."""

    async def _run(*cmd: str, timeout: float = 15.0):
        return code, out, err

    return _run


# -- pure helpers ------------------------------------------------------------


def test_fail_reason_skips_noise():
    # A version hint and a uv warning shouldn't masquerade as the failure reason.
    err = "warning: UV_NATIVE_TLS is deprecated\nHint: a new version is available\nNot logged in"
    assert auth_check._fail_reason(1, "", err) == "Not logged in"


def test_fail_reason_falls_back_to_exit_code():
    assert auth_check._fail_reason(2, "", "") == "exit 2"


def test_status_line_marks_and_trims():
    assert auth_check.Status("Modal", True, "workspace acme").line() == "  ✅ Modal              workspace acme"
    assert auth_check.Status("Modal", True, "").line() == "  ✅ Modal"
    assert auth_check.Status("Modal", False, "nope").line().startswith("  ❌")


# -- per-provider probes -----------------------------------------------------


def test_modal_reports_workspace_without_id(monkeypatch):
    monkeypatch.setattr(auth_check, "_run", fake_run(0, "Workspace: acme-corp (ac-1a2b)\nUser: someone"))
    monkeypatch.setenv("MINI_NO_PROJECT_CONFIG", "1")
    monkeypatch.delenv("MODAL_ENVIRONMENT", raising=False)
    status = asyncio.run(auth_check.check_modal())
    assert status.ok and status.detail == "workspace acme-corp"
    # A profile's Environment is part of "where this session writes", so it's shown beside the workspace.
    monkeypatch.setenv("MODAL_ENVIRONMENT", "dev")
    assert asyncio.run(auth_check.check_modal()).detail == "workspace acme-corp, environment dev"


def test_a_failed_probe_says_why(monkeypatch):
    """Modal quotes the tool's own message; Hugging Face answers with the fix instead."""
    monkeypatch.setattr(auth_check, "_run", fake_run(1, "", "Token missing"))
    assert asyncio.run(auth_check.check_modal()) == auth_check.Status("Modal", False, "Token missing")
    monkeypatch.setattr(auth_check, "_token_scope", fake_scope(None))
    assert asyncio.run(auth_check.check_hf()) == auth_check.Status(
        "Hugging Face", False, "not logged in — run ./go auth"
    )


@pytest.mark.parametrize(
    ("bucket", "repo", "profile", "detail"),
    [
        ("octocat/data-store", None, None, "user octocat, bucket octocat/data-store"),
        (
            "octocat/data-store",
            "octocat/pub",
            None,
            "user octocat, bucket octocat/data-store, dataset octocat/pub",
        ),
        (None, None, None, "user octocat, no store-bucket set"),
        ("octocat/dev-store", None, "dev", "user octocat, profile dev, bucket octocat/dev-store"),
    ],
    ids=["publish-tier-off-is-not-shown", "publish-repo-set", "no-bucket", "profile-named-when-active"],
)
def test_hf_reports_the_user_and_the_configured_repos(monkeypatch, bucket, repo, profile, detail):
    monkeypatch.setattr(auth_check, "_run", fake_run(0, "user=octocat"))
    monkeypatch.setattr(auth_check, "_token_scope", fake_scope(None))
    monkeypatch.setattr("mini.store.store_bucket", lambda: bucket)
    monkeypatch.setattr("mini.store.publish_repo", lambda: repo)
    monkeypatch.setattr("mini.store.active_profile", lambda: profile)
    assert asyncio.run(auth_check.check_hf()) == auth_check.Status("Hugging Face", True, detail)


def test_github_extracts_account(monkeypatch):
    monkeypatch.setattr(auth_check, "_run", fake_run(0, "", "✓ Logged in to github.com account octocat (keyring)"))
    status = asyncio.run(auth_check.check_github())
    assert status.ok and status.detail == "account octocat"


def test_missing_binary_reports_not_installed(monkeypatch):
    monkeypatch.setattr(auth_check, "_run", fake_run(127, "", "not installed"))
    status = asyncio.run(auth_check.check_github())
    assert not status.ok and status.detail == "not installed"


# -- environment-aware selection ---------------------------------------------


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, ["check_modal", "check_hf", "check_github", "check_claude"]),
        # On Claude Code for the web GitHub goes through MCP tools, not `gh`.
        ({"CLAUDE_CODE_REMOTE": "true"}, ["check_modal", "check_hf", "check_claude"]),
        # Claude's own auth is irrelevant when Claude is the caller.
        ({"CLAUDECODE": "1"}, ["check_modal", "check_hf", "check_github"]),
        ({"CLAUDE_CODE_REMOTE": "true", "CLAUDECODE": "1"}, ["check_modal", "check_hf"]),
    ],
    ids=["a-local-shell", "the-web", "claude-is-the-caller", "a-web-agent"],
)
def test_the_environment_selects_the_checks(monkeypatch, env: dict[str, str], expected: list[str]):
    for name in ("CLAUDE_CODE_REMOTE", "CLAUDECODE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert [check.__name__ for check in auth_check._relevant_checks()] == expected


# -- which storage pairs the token can reach ---------------------------------


def fine(**grants: str) -> dict:
    """An access-token record granting `write` or `read` on each named repo, as the Hub reports it."""
    perms = {"write": ["repo.access.read", "repo.content.read", "repo.write"], "read": ["repo.content.read"]}
    return {
        "fineGrained": {
            "global": [],
            "scoped": [
                {"entity": {"type": "bucket", "name": repo}, "permissions": perms[level]}
                for repo, level in grants.items()
            ],
        }
    }


PROD = ["acme/store", "acme/pub"]
DEV = ["acme/store-dev", "acme/pub-dev"]


def test_a_pair_is_writable_only_when_both_halves_are():
    token = fine(**{"acme/store": "write", "acme/pub": "read"})
    assert auth_check._pair_access(token, PROD) == "read only"
    assert auth_check._pair_access(fine(**dict.fromkeys(PROD, "write")), PROD) == "write"


def test_an_unnamed_pair_is_out_of_reach():
    assert auth_check._pair_access(fine(**dict.fromkeys(DEV, "write")), PROD) == "none"


def test_half_a_pair_is_partial():
    assert auth_check._pair_access(fine(**{"acme/store": "write"}), PROD) == "partial"


def test_a_namespace_grant_covers_its_repos():
    token = {
        "fineGrained": {
            "global": [],
            "scoped": [{"entity": {"type": "user", "name": "acme"}, "permissions": ["repo.write"]}],
        }
    }
    assert auth_check._pair_access(token, PROD) == "write"


def test_a_classic_token_carries_one_role_everywhere():
    assert auth_check._pair_access({"role": "write"}, PROD) == "write"
    assert auth_check._pair_access({"role": "read"}, PROD) == "read only"
    assert auth_check._pair_access({}, PROD) == "none"


@pytest.mark.parametrize(
    ("grants", "expected"),
    [
        ({**dict.fromkeys(PROD, "write"), **dict.fromkeys(DEV, "write")}, "token: write on production and dev"),
        (dict.fromkeys(DEV, "write"), "token: write on dev; no access to production"),
        (
            {**dict.fromkeys(PROD, "read"), **dict.fromkeys(DEV, "write")},
            "token: write on dev; read only on production",
        ),
    ],
    ids=["both", "dev-only", "mixed"],
)
def test_the_scope_line_groups_pairs_by_level(monkeypatch, grants, expected):
    monkeypatch.setattr(auth_check, "_configured_pairs", lambda: [("production", PROD), ("dev", DEV)])

    async def _whoami():
        return {"auth": {"accessToken": fine(**grants)}}

    monkeypatch.setattr(auth_check, "_whoami", _whoami)
    assert asyncio.run(auth_check._token_scope()) == expected


def test_the_scope_line_is_dropped_when_the_lookup_fails(monkeypatch):
    monkeypatch.setattr(auth_check, "_configured_pairs", lambda: [("production", PROD)])

    async def _whoami():
        raise RuntimeError("no network")

    monkeypatch.setattr(auth_check, "_whoami", _whoami)
    assert asyncio.run(auth_check._token_scope()) is None


def test_the_scope_line_lands_under_the_detail(monkeypatch):
    monkeypatch.setattr(auth_check, "_run", fake_run(0, "user=octocat"))
    monkeypatch.setattr(auth_check, "_token_scope", fake_scope("token: write on dev"))
    monkeypatch.setattr("mini.store.store_bucket", lambda: "acme/store-dev")
    monkeypatch.setattr("mini.store.publish_repo", lambda: None)
    monkeypatch.setattr("mini.store.active_profile", lambda: "dev")
    head, note = asyncio.run(auth_check.check_hf()).line().splitlines()
    # The mark is one character but two terminal columns, so the note sits one character
    # past where `detail` starts and the two line up on screen.
    assert note.index("token: write on dev") == head.index("user octocat") + 1

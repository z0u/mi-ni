"""Tests for the credential status probe (`./go auth --check`).

The parsing helpers are pure; the per-provider checks shell out through a single
`_run`, so we drive them by swapping in a fake that returns canned tool output.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "auth_check", Path(__file__).resolve().parent.parent / "scripts" / "auth_check.py"
)
assert _SPEC and _SPEC.loader
auth_check = importlib.util.module_from_spec(_SPEC)
sys.modules["auth_check"] = auth_check  # so @dataclass can resolve the module by name
_SPEC.loader.exec_module(auth_check)


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
    status = asyncio.run(auth_check.check_modal())
    assert status.ok and status.detail == "workspace acme-corp"


def test_modal_failure_surfaces_reason(monkeypatch):
    monkeypatch.setattr(auth_check, "_run", fake_run(1, "", "Token missing"))
    status = asyncio.run(auth_check.check_modal())
    assert not status.ok and status.detail == "Token missing"


def test_hf_includes_user_and_bucket(monkeypatch):
    monkeypatch.setattr(auth_check, "_run", fake_run(0, "user=octocat"))
    monkeypatch.setattr("mini.store.store_bucket", lambda: "octocat/data-store")
    monkeypatch.setattr("mini.store.publish_repo", lambda: None)  # publish tier off → not shown
    status = asyncio.run(auth_check.check_hf())
    assert status.ok and status.detail == "user octocat, bucket octocat/data-store"


def test_hf_shows_publish_repo_when_set(monkeypatch):
    monkeypatch.setattr(auth_check, "_run", fake_run(0, "user=octocat"))
    monkeypatch.setattr("mini.store.store_bucket", lambda: "octocat/data-store")
    monkeypatch.setattr("mini.store.publish_repo", lambda: "octocat/pub")
    status = asyncio.run(auth_check.check_hf())
    assert status.ok and status.detail == "user octocat, bucket octocat/data-store, publish-repo octocat/pub"


def test_hf_notes_missing_bucket(monkeypatch):
    monkeypatch.setattr(auth_check, "_run", fake_run(0, "user=octocat"))
    monkeypatch.setattr("mini.store.store_bucket", lambda: None)
    monkeypatch.setattr("mini.store.publish_repo", lambda: None)
    status = asyncio.run(auth_check.check_hf())
    assert status.ok and "no store-bucket set" in status.detail


def test_hf_not_logged_in(monkeypatch):
    monkeypatch.setattr(auth_check, "_run", fake_run(1, "", "Not logged in"))
    status = asyncio.run(auth_check.check_hf())
    assert not status.ok


def test_wandb_reads_username_from_login(monkeypatch):
    err = "wandb: Currently logged in as: acme to https://api.wandb.ai. Use `wandb login --relogin` to force relogin"
    monkeypatch.setattr(auth_check, "_run", fake_run(0, "", err))
    status = asyncio.run(auth_check.check_wandb())
    assert status.ok and status.detail == "user acme"


def test_wandb_not_logged_in_is_failure(monkeypatch):
    monkeypatch.setattr(auth_check, "_run", fake_run(1, "", "Error: No API key configured."))
    status = asyncio.run(auth_check.check_wandb())
    assert not status.ok and "no API key" in status.detail


def test_github_extracts_account(monkeypatch):
    monkeypatch.setattr(auth_check, "_run", fake_run(0, "", "✓ Logged in to github.com account octocat (keyring)"))
    status = asyncio.run(auth_check.check_github())
    assert status.ok and status.detail == "account octocat"


def test_missing_binary_reports_not_installed(monkeypatch):
    monkeypatch.setattr(auth_check, "_run", fake_run(127, "", "not installed"))
    status = asyncio.run(auth_check.check_github())
    assert not status.ok and status.detail == "not installed"


# -- environment-aware selection ---------------------------------------------


def test_all_checks_run_by_default(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_REMOTE", raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    assert auth_check._relevant_checks() == [
        auth_check.check_modal,
        auth_check.check_hf,
        auth_check.check_wandb,
        auth_check.check_github,
        auth_check.check_claude,
    ]


def test_github_skipped_on_the_web(monkeypatch):
    # On Claude Code for the web GitHub goes through MCP tools, not `gh`.
    monkeypatch.setenv("CLAUDE_CODE_REMOTE", "true")
    monkeypatch.delenv("CLAUDECODE", raising=False)
    assert auth_check.check_github not in auth_check._relevant_checks()
    assert auth_check.check_claude in auth_check._relevant_checks()


def test_claude_check_skipped_when_claude_is_the_caller(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_REMOTE", raising=False)
    monkeypatch.setenv("CLAUDECODE", "1")
    assert auth_check.check_claude not in auth_check._relevant_checks()
    assert auth_check.check_github in auth_check._relevant_checks()


def test_web_agent_runs_only_service_checks(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_REMOTE", "true")
    monkeypatch.setenv("CLAUDECODE", "1")
    assert auth_check._relevant_checks() == [
        auth_check.check_modal,
        auth_check.check_hf,
        auth_check.check_wandb,
    ]

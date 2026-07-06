#!/usr/bin/env python
"""Fast, non-interactive credential status check (`./go auth --check`).

Agents (and humans) often don't realise Modal and Hugging Face are already
authenticated in a fresh shell — and poking at the raw tools to find out can
spill a token into a transcript. This runs each provider's real CLI
concurrently and reports only whether the credential works plus safe metadata
(workspace, bucket, user); never the secret itself.

Every probe runs with stdin closed and a timeout, so an unauthenticated tool
reports "not logged in" instead of blocking on a prompt.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from asyncio.subprocess import DEVNULL, PIPE
from collections.abc import Callable, Coroutine
from typing import Any
from dataclasses import dataclass


@dataclass
class Status:
    label: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        mark = "✅" if self.ok else "❌"
        return f"  {mark} {self.label:<18} {self.detail}".rstrip()


async def _run(*cmd: str, timeout: float = 15.0) -> tuple[int, str, str]:
    """Run *cmd*, returning ``(returncode, stdout, stderr)``.

    Closes stdin so a tool that would prompt fails fast, and kills the child on
    timeout. A missing binary is reported as code 127 (like a shell would).
    """
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdin=DEVNULL, stdout=PIPE, stderr=PIPE)
    except FileNotFoundError, PermissionError:
        return 127, "", "not installed"
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return -1, "", f"timed out after {timeout:g}s"
    return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def _fail_reason(code: int, out: str, err: str) -> str:
    """A short reason for a failed probe, preferring the tool's own message."""
    if err.strip() == "not installed":
        return "not installed"
    # hf prints a version "Hint:" to stderr even on success — skip those.
    for line in (err + "\n" + out).splitlines():
        line = line.strip()
        if line and not line.lower().startswith("hint:") and "warning:" not in line.lower():
            return line
    return f"exit {code}"


# -- per-provider probes -----------------------------------------------------
#
# Each returns a Status. The token itself is never included in `detail`.


async def check_modal() -> Status:
    code, out, err = await _run("modal", "token", "info")
    if code != 0:
        return Status("Modal", False, _fail_reason(code, out, err))
    # The line reads "Workspace: <name> (<internal-id>)"; keep the name, drop the id.
    match = re.search(r"^\s*Workspace:\s*(\S+)", out, re.MULTILINE)
    workspace = match.group(1) if match else ""
    return Status("Modal", True, f"workspace {workspace}" if workspace else "authenticated")


async def check_hf() -> Status:
    from mini.store import publish_repo, store_bucket

    code, out, err = await _run("hf", "auth", "whoami")
    text = out + err
    if code != 0 or "not logged in" in text.lower():
        return Status("Hugging Face", False, "not logged in — run ./go auth")
    # `hf auth whoami` prints `user=<name>`; fall back to the first plain line.
    match = re.search(r"^\s*user[=:]\s*(\S+)", out, re.MULTILINE | re.IGNORECASE)
    user = match.group(1) if match else next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    bucket, repo = store_bucket(), publish_repo()
    parts = [
        p
        for p in (
            f"user {user}" if user else "",
            f"bucket {bucket}" if bucket else "no store-bucket set",
            # Only shown when set — the publish tier is opt-in (#38); unset means publish stays in the bucket.
            f"publish-repo {repo}" if repo else "",
        )
        if p
    ]
    return Status("Hugging Face", True, ", ".join(parts))


async def check_wandb() -> Status:
    # `wandb status` reports the settings-file api_key only — it misses the common
    # case of a netrc- or WANDB_API_KEY-based login. `wandb login` (no key argument,
    # stdin closed) checks all three sources and exits non-zero with no TTY if none
    # are set, so it's a more reliable — and still read-only — probe.
    code, out, err = await _run("wandb", "login")
    if code != 0:
        return Status("WandB", False, "no API key — run ./go auth")
    match = re.search(r"Currently logged in as:\s*(\S+)", out + err)
    return Status("WandB", True, f"user {match.group(1)}" if match else "authenticated")


async def check_github() -> Status:
    code, out, err = await _run("gh", "auth", "status")
    text = out + err
    if code != 0:
        return Status("GitHub", False, "not installed" if "not installed" in err else "not logged in — run ./go auth")
    match = re.search(r"account (\S+)", text)
    return Status("GitHub", True, f"account {match.group(1)}" if match else "authenticated")


async def check_claude() -> Status:
    code, out, err = await _run("claude", "auth", "status")
    if code != 0:
        return Status(
            "Claude Code", False, "not installed" if "not installed" in err else "not logged in — run ./go auth"
        )
    return Status("Claude Code", True, "authenticated")


def _relevant_checks() -> list[Callable[[], Coroutine[Any, Any, Status]]]:
    """The probes worth running in this environment.

    Modal and Hugging Face (the resources agents miss) always run. The other two
    are context-dependent:

    - Skip GitHub on Claude Code for the web (``CLAUDE_CODE_REMOTE``): there GitHub
      is reached through the MCP tools, ``gh`` isn't installed, and the network
      policy blocks its API — so a ❌ would be noise, not signal.
    - Skip the Claude Code check when Claude itself is the caller (``CLAUDECODE``):
      its own auth is irrelevant to the run.
    """
    checks: list[Callable[[], Coroutine[Any, Any, Status]]] = [check_modal, check_hf, check_wandb]
    if os.environ.get("CLAUDE_CODE_REMOTE") != "true":
        checks.append(check_github)
    if not os.environ.get("CLAUDECODE"):
        checks.append(check_claude)
    return checks


async def _gather() -> list[Status]:
    return list(await asyncio.gather(*(check() for check in _relevant_checks())))


def main() -> int:
    print("Checking credentials…\n", file=sys.stderr)
    statuses = asyncio.run(_gather())
    for status in statuses:
        print(status.line())
    return 0 if all(s.ok for s in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())

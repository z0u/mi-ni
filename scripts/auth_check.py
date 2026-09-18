#!/usr/bin/env python
"""Fast, non-interactive credential status check (`./go auth --check`).

Agents (and humans) often don't realise Modal and Hugging Face are already authenticated in a fresh shell — and poking at the raw tools to find out can spill a token into a transcript. This runs each provider's real CLI concurrently and reports only whether the credential works plus safe metadata (workspace, bucket, user); never the secret itself.

Every probe runs with stdin closed and a timeout, so an unauthenticated tool reports "not logged in" instead of blocking on a prompt.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from asyncio.subprocess import DEVNULL, PIPE
from collections.abc import Callable, Coroutine
from typing import Any
from dataclasses import dataclass, field


@dataclass
class Status:
    label: str
    ok: bool
    detail: str = ""
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        mark = "✅" if self.ok else "❌"
        head = f"  {mark} {self.label:<18} {self.detail}".rstrip()
        # Continuation lines align under `detail` as a terminal lays it out: two spaces,
        # the mark (one character, two columns), a space, the padded label, a space.
        indent = " " * (2 + 2 + 1 + 18 + 1)
        return "\n".join([head, *(f"{indent}{n}" for n in self.notes)])


async def _run(*cmd: str, timeout: float = 15.0) -> tuple[int, str, str]:
    """Run *cmd*, returning ``(returncode, stdout, stderr)``.

    Closes stdin so a tool that would prompt fails fast, and kills the child on timeout. A missing binary is reported as code 127 (like a shell would).
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
    # Only shown when set — unset is Modal's default Environment, the production case.
    from mini.store import modal_environment

    parts = [f"workspace {workspace}" if workspace else "authenticated"]
    if env := modal_environment():
        parts.append(f"environment {env}")
    return Status("Modal", True, ", ".join(parts))


# -- which storage pairs the token can reach ---------------------------------
#
# `store-bucket` / `publish-repo` name one pair per profile, but the token decides
# which of them can be written (see eng/environments.md). Reading the names alone
# can't tell a sandbox credential from one that also reaches production, so the
# check reports the token's own grants. They come from `whoami()`, which returns
# the scope list for a fine-grained token: no write is attempted, and no part of
# the token appears in the output.

# What a Hub grant list has to contain for each level, most-privileged first.
_REPO_PERMS = (("write", {"repo.write"}), ("read only", {"repo.content.read", "repo.access.read"}))


def _configured_pairs() -> list[tuple[str, list[str]]]:
    """Each configured storage pair as ``(label, repo ids)`` — production first, then each profile."""
    from mini.store import profiles, publish_repo, store_bucket

    pairs = []
    for profile in (None, *profiles()):
        repos = [r for r in (store_bucket(profile=profile), publish_repo(profile=profile)) if r]
        if repos:
            pairs.append((profile or "production", repos))
    return pairs


def _repo_access(access_token: dict, repo: str) -> str:
    """The level *access_token* grants on *repo* (``namespace/name``): ``write``, ``read only``, or ``""``."""
    fine = access_token.get("fineGrained")
    if fine is None:
        # A classic token carries one role over everything the account can see.
        return {"write": "write", "read": "read only"}.get(access_token.get("role", ""), "")
    namespace = repo.split("/")[0]
    granted: set[str] = {p for p in fine.get("global", []) if p.startswith("repo.")}
    for scope in fine.get("scoped", []):
        entity = scope.get("entity", {})
        # A repo-scoped grant names the repo; a user- or org-scoped one covers the namespace.
        if entity.get("name") in (repo, namespace if entity.get("type") in ("user", "org") else None):
            granted |= set(scope.get("permissions", []))
    return next((level for level, need in _REPO_PERMS if granted & need), "")


def _pair_access(access_token: dict, repos: list[str]) -> str:
    """The level the whole pair shares — a pair is only writable if both halves are."""
    levels = [_repo_access(access_token, r) for r in repos]
    if all(lv == "write" for lv in levels):
        return "write"
    if all(levels):
        return "read only"
    return "partial" if any(levels) else "none"


async def _whoami() -> dict:
    """The Hub's account record for the active token. Its own function so a test can stand in for it."""
    from huggingface_hub import HfApi

    return await asyncio.wait_for(asyncio.to_thread(HfApi().whoami), 15.0)


async def _token_scope() -> str | None:
    """One line naming what the Hugging Face token can do to each configured pair, or ``None`` if unknown."""
    try:
        pairs = _configured_pairs()
        if not pairs:
            return None
        access_token = (await _whoami()).get("auth", {}).get("accessToken") or {}
        by_level: dict[str, list[str]] = {}
        for label, repos in pairs:
            by_level.setdefault(_pair_access(access_token, repos), []).append(label)
    except Exception:
        # Advisory detail only — a check that already knows the token works shouldn't
        # fail because the scope lookup didn't.
        return None
    # Ordered most-privileged first, each with the phrasing that reads as a sentence.
    phrasing = [
        ("write", "write on"),
        ("read only", "read only on"),
        ("partial", "partial on"),
        ("none", "no access to"),
    ]
    return "token: " + "; ".join(f"{say} {_and_list(by_level[lv])}" for lv, say in phrasing if lv in by_level)


def _and_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


async def check_hf() -> Status:
    from mini.store import active_profile, publish_repo, store_bucket

    # The CLI probe answers "is this shell authenticated"; the scope lookup answers
    # "what can it write". Run them together so the second costs no wall clock.
    (code, out, err), scope = await asyncio.gather(_run("hf", "auth", "whoami"), _token_scope())
    text = out + err
    if code != 0 or "not logged in" in text.lower():
        return Status("Hugging Face", False, "not logged in — run ./go auth")
    # `hf auth whoami` prints `user=<name>`; fall back to the first plain line.
    match = re.search(r"^\s*user[=:]\s*(\S+)", out, re.MULTILINE | re.IGNORECASE)
    user = match.group(1) if match else next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    bucket, repo, profile = store_bucket(), publish_repo(), active_profile()
    parts = [
        p
        for p in (
            f"user {user}" if user else "",
            # Only shown when set — unset is the base (production) pair, the usual case.
            f"profile {profile}" if profile else "",
            f"bucket {bucket}" if bucket else "no store-bucket set",
            # Only shown when set — the publish tier is opt-in (#38); unset means publish stays in the bucket.
            f"dataset {repo}" if repo else "",
        )
        if p
    ]
    return Status("Hugging Face", True, ", ".join(parts), notes=[scope] if scope else [])


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

    Modal and Hugging Face (the resources agents miss) always run. The other two are context-dependent:

    - Skip GitHub on Claude Code for the web (``CLAUDE_CODE_REMOTE``): there GitHub is reached through the MCP tools, ``gh`` isn't installed, and the network policy blocks its API — so a ❌ would be noise, not signal.
    - Skip the Claude Code check when Claude itself is the caller (``CLAUDECODE``): its own auth is irrelevant to the run.
    """
    checks: list[Callable[[], Coroutine[Any, Any, Status]]] = [check_modal, check_hf]
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

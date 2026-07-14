"""
Run lineage: enough provenance to reproduce (or forensically reconstruct) a run.

A memoized run is driven across many short-lived processes; this module captures,
once per driver wake, the things a report needs to answer *what produced this, on
what, by whom, and can I recreate it exactly*:

- the **code state** — git sha, branch, tags pointing at HEAD, remote(s), and the
  working-tree diff when the tree is dirty (so a run off uncommitted code is still
  reconstructable);
- **who drove it** — the AI agent(s) and a non-PII operator handle (the repo owner
  from the git remote; the git ``user.name`` is a bot in agent/CI contexts and a
  real name is PII, so it's deliberately not used);
- the **environment that spawned/managed the work** — this driver process;
- **when** — captured, and first-captured across wakes.

It's deliberately dependency-light (``subprocess`` + stdlib) so it runs identically
in a notebook, the CLI driver, a CI runner, or a cloud sandbox, and never drags a
heavy import into the hot control plane. Per-*task* execution facts (the Modal
container / GPU / RAM a step actually ran on) are captured separately by the worker
(:func:`mini.runs.compute_env`); Modal **cost** is reconciled post-run from the
billing API (:func:`mini.modal_apparatus.query_cost`).

Secrets never enter a record: remote URLs are stripped of embedded credentials, and
only a safe allowlist of environment markers (versions, session kinds — never
tokens, emails, or account ids) is recorded.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "run_lineage",
    "git_lineage",
    "agents",
    "operators",
    "driver_env",
    "merge_run_lineage",
    "upstream_snapshot",
]

# A dirty tree's diff rides the hot control-plane record, so cap it — a giant
# generated-file churn shouldn't bloat every poll. The flag says we truncated.
_DIFF_CAP = 200_000  # bytes

# Markers identifying the project root (mirrors mini.runs / mini.reports).
_ROOT_MARKERS = ("pyproject.toml", ".git")

# ``scheme://user:pass@host/…`` — strip the userinfo so a tokened/proxied remote
# URL (``http://local_proxy@…`` / ``https://x-access-token:ghs_…@github.com``)
# never lands in a record.
_USERINFO = re.compile(r"(?<=://)[^/@]*@")

# What we keep out of a function/agent name in a Modal-safe, log-safe id.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _project_root(start: Path | None = None) -> Path:
    """Nearest ancestor with a ``pyproject.toml`` / ``.git`` (else *start*)."""
    start = (start or Path.cwd()).resolve()
    for d in (start, *start.parents):
        if any((d / m).exists() for m in _ROOT_MARKERS):
            return d
    return start


def _git(root: Path, *args: str) -> str | None:
    """Run one ``git`` command under *root*; stripped stdout, or ``None`` on any failure.

    Never raises — a missing git binary, a non-repo dir, or a non-zero exit all
    read as "unknown", so lineage capture degrades gracefully instead of taking a
    run down.
    """
    if not shutil.which("git"):
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _sanitize_url(url: str) -> str:
    """Drop any ``user:pass@`` userinfo from a remote URL (keep host/path)."""
    return _USERINFO.sub("", url)


def _git_remotes(root: Path) -> dict[str, str]:
    """Named remotes with credentials stripped from their URLs."""
    remotes: dict[str, str] = {}
    for line in (_git(root, "remote", "-v") or "").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            remotes.setdefault(parts[0], _sanitize_url(parts[1]))
    return remotes


def _git_worktree(root: Path, info: dict[str, Any]) -> None:
    """Fill *info* with dirty state: the capped tracked-file diff + untracked paths."""
    status = _git(root, "status", "--porcelain") or ""
    info["dirty"] = bool(status)
    if not status:
        return
    if diff := _git(root, "diff", "HEAD"):
        info["diff_truncated"] = len(diff.encode()) > _DIFF_CAP
        info["diff"] = diff[:_DIFF_CAP]
    if untracked := [line[3:] for line in status.splitlines() if line.startswith("??")]:
        info["untracked"] = untracked


def git_lineage(root: Path | str | None = None) -> dict[str, Any] | None:
    """The repository state a run was launched from, or ``None`` when it's not a git repo.

    Captures sha + branch + tags-at-HEAD + ``describe`` + sanitized remotes + the
    last commit's subject/date, and — when the tree is dirty — the tracked-file diff
    (capped) and the list of untracked *paths* (names only; their contents may be
    huge or secret). This is the "recreate it exactly" half: a clean tree pins to a
    sha, a dirty one carries the delta on top of it.
    """
    root = Path(root) if root else _project_root()
    sha = _git(root, "rev-parse", "HEAD")
    if sha is None:
        return None
    info: dict[str, Any] = {"sha": sha, "short_sha": sha[:12]}
    if (branch := _git(root, "rev-parse", "--abbrev-ref", "HEAD")) and branch != "HEAD":
        info["branch"] = branch
    if desc := _git(root, "describe", "--tags", "--always", "--dirty"):
        info["describe"] = desc
    if tags := _git(root, "tag", "--points-at", "HEAD"):
        info["tags"] = tags.split("\n")
    if remotes := _git_remotes(root):
        info["remotes"] = remotes
    if subject := _git(root, "log", "-1", "--pretty=%s"):
        info["subject"] = subject
    if committed_at := _git(root, "log", "-1", "--pretty=%cI"):
        info["committed_at"] = committed_at
    _git_worktree(root, info)
    return info


# Each entry: a probe over ``os.environ`` and the fields to lift if it fires. Values
# are env-var names (looked up at capture time) — only versions / entrypoints /
# session-kinds, never tokens, emails, or account ids.
_AGENT_SPECS: tuple[tuple[str, tuple[str, ...], dict[str, str]], ...] = (
    (
        "claude-code",
        ("CLAUDECODE", "CLAUDE_CODE_VERSION", "CLAUDE_CODE_ENTRYPOINT"),
        {"version": "CLAUDE_CODE_VERSION", "entrypoint": "CLAUDE_CODE_ENTRYPOINT"},
    ),
    ("openai-codex", ("CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED"), {}),
    ("cursor", ("CURSOR_TRACE_ID", "CURSOR_AGENT"), {}),
    ("aider", ("AIDER_MODEL", "AIDER_VERSION"), {"version": "AIDER_VERSION", "model": "AIDER_MODEL"}),
    ("github-copilot", ("COPILOT_AGENT_ID", "GITHUB_COPILOT_AGENT"), {}),
)


def agents(env: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Detected AI agents driving the run (non-PII: name + version/entrypoint only).

    Keyed off well-known environment markers each tool sets. An unknown agent goes
    undetected rather than guessed; the human operator is captured separately by
    :func:`operators`.
    """
    env = env if env is not None else dict(os.environ)
    out: list[dict[str, str]] = []
    for name, probes, fields in _AGENT_SPECS:
        if not any(env.get(p) for p in probes):
            continue
        agent = {"name": name}
        for label, var in fields.items():
            if val := env.get(var):
                agent[label] = val
        out.append(agent)
    return out


# The owner/repo tail of a remote URL: ``…/z0u/mi-ni(.git)`` or scp ``git@host:z0u/mi-ni``.
_REMOTE_OWNER = re.compile(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?/?$")


def operators(root: Path | str | None = None) -> list[dict[str, str]]:
    """Non-PII handle(s) for *who* ran it — the git host owner from the remote(s).

    We deliberately **don't** use the git-configured ``user.name`` / ``user.email``:
    in an agent or CI context that's a bot identity (e.g. ``Claude
    <noreply@anthropic.com>``), not the operator, and a real name/email is PII we
    won't put in a record that may end up in a *published* report. The clean signal
    that's actually available offline is the repo owner in the remote URL (e.g.
    ``z0u`` from ``github.com/z0u/mi-ni``) — a public pseudonym, useful for forensics
    and safe to persist.

    The true operator identity (the human behind a Claude Code / CI session) lives
    only in PII-bearing or sensitive env (session email, account/org ids, session
    tokens), so it's intentionally left uncaptured; resolving a token to its login
    would need a network call and often returns a bot anyway.
    """
    handles: list[str] = []
    for url in _git_remotes(Path(root) if root else _project_root()).values():
        if m := _REMOTE_OWNER.search(url):
            handles.append(m.group(1))
    return [{"handle": h, "source": "git-remote"} for h in dict.fromkeys(handles)]


def _detect_runner(env: dict[str, str]) -> dict[str, str] | None:
    """The kind of environment that spawned/managed this driver, with safe markers.

    Only non-secret identifiers (workflow/run/ref, session kind) — never tokens or
    account ids. ``None`` when nothing distinctive is set (a plain local shell).
    """
    if env.get("CLAUDE_CODE_REMOTE"):
        runner = {"kind": "claude-code-remote"}
        if t := env.get("CLAUDE_CODE_REMOTE_ENVIRONMENT_TYPE"):
            runner["environment_type"] = t
        if v := env.get("CLAUDE_CODE_ENVIRONMENT_RUNNER_VERSION"):
            runner["runner_version"] = v
        return runner
    if env.get("GITHUB_ACTIONS"):
        return {
            "kind": "github-actions",
            **{
                k: v
                for k, v in (
                    ("workflow", env.get("GITHUB_WORKFLOW")),
                    ("run_id", env.get("GITHUB_RUN_ID")),
                    ("ref", env.get("GITHUB_REF")),
                )
                if v
            },
        }
    if env.get("REMOTE_CONTAINERS") or env.get("DEVCONTAINER") or env.get("CODESPACES"):
        return {"kind": "devcontainer"}
    if env.get("CI"):
        return {"kind": "ci"}
    return None


def driver_env(env: dict[str, str] | None = None) -> dict[str, Any]:
    """The environment that spawned and managed the work (this driver process).

    Distinct from the per-task *execution* environment (a Modal container) captured
    by :func:`mini.runs.compute_env`: this is where ``main(ctx)`` is ticked and
    tasks are launched from.
    """
    env = env if env is not None else dict(os.environ)
    info: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if runner := _detect_runner(env):
        info["runner"] = runner
    return info


def run_lineage(root: Path | str | None = None) -> dict[str, Any]:
    """Assemble the full run-level lineage snapshot for this driver wake."""
    root = Path(root) if root else _project_root()
    now = datetime.now(timezone.utc)
    lineage: dict[str, Any] = {
        "captured_at": now.isoformat(),
        "captured_at_epoch": now.timestamp(),
        "agents": agents(),
        "operators": operators(root),
        "driver": driver_env(),
    }
    if git := git_lineage(root):
        lineage["git"] = git
    return lineage


def merge_run_lineage(prev: dict[str, Any] | None, fresh: dict[str, Any]) -> dict[str, Any]:
    """Fold a *fresh* capture over *prev*, keeping first-run breadcrumbs.

    A detached run is ticked over many wakes; the meaningful code state is the
    *latest* one (edits re-run tasks, so the final git state is what produced the
    current results), but the run's *start* and how many times it woke are worth
    keeping. Last-writer-wins on the snapshot, first-writer-wins on the origin.
    Already-captured upstreams survive a wake that didn't re-resolve them.
    """
    merged = dict(fresh)
    if prev:
        merged["first_captured_at"] = prev.get("first_captured_at", prev.get("captured_at"))
        merged["first_captured_at_epoch"] = prev.get("first_captured_at_epoch", prev.get("captured_at_epoch"))
        merged["wakes"] = (prev.get("wakes") or 1) + 1
        if prev.get("upstreams") and not merged.get("upstreams"):
            merged["upstreams"] = prev["upstreams"]
    else:
        merged["first_captured_at"] = fresh.get("captured_at")
        merged["first_captured_at_epoch"] = fresh.get("captured_at_epoch")
        merged["wakes"] = 1
    return merged


def upstream_snapshot(name: str, meta: dict[str, Any]) -> dict[str, Any]:
    """A compact provenance record of an upstream experiment, for embedding downstream.

    Given experiment *A*'s run *meta* (its stored ``lineage`` + Modal app ids), pull
    just enough to trace back to it — its name, code state, when it first ran, and
    the Modal apps that produced it — so a report for *B* can prove which *A* it was
    built on without carrying *A*'s whole record.
    """
    lineage = meta.get("lineage") or {}
    git = lineage.get("git") or {}
    snap: dict[str, Any] = {"experiment": name}
    if sha := git.get("sha"):
        snap["git_sha"] = sha
    if describe := git.get("describe"):
        snap["git_describe"] = describe
    if git.get("dirty"):
        snap["git_dirty"] = True
    if run_at := lineage.get("first_captured_at"):
        snap["run_at"] = run_at
    if app_ids := meta.get("modal_app_ids"):
        snap["modal_app_ids"] = list(app_ids)
    return snap

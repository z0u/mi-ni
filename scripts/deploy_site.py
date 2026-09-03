#!/usr/bin/env python
"""Build the whole site and deploy it as one commit.

The site is a function of two things: `main`, which production is built from, and the set of open PRs, each of which gets a preview under `pr-preview/pr-<n>/`. So every run builds all of it from scratch and force-pushes the result to `gh-pages` as a single parentless commit. Nothing on the branch carries over from one run to the next, and nothing has to be torn down: a closed PR is absent from the next build, whatever event (or none) its close produced.

**Why reconcile rather than react.** The arrangement this replaced had three writers of `gh-pages` — a production deploy on push to `main`, a preview deploy per PR event, and a teardown on close — each writing one slice of the branch and leaving the rest alone. Every guard it grew was about the other writers: a lease on the history prune, `force: false` on the deploy, `clean-exclude` on the umbrella directory, and finally a sweep for the previews whose teardown had been overwritten by an in-flight build or never scheduled at all (seven of them by 2026-08-31, two thirds of what the site served). A run that rebuilds everything from current state has no other writers to guard against, and it makes the workflow's single concurrency group safe: a pending run displaced by a newer arrival loses nothing, because the newer run reads the same state, later. `eng/publishing.md` has the longer form.

**What it costs** is rebuilding every preview on every event, at about 30 s each. This repository has a handful of PRs open at a time, so a run is a minute or two. A preview that fails to build is reported (a warning, and a note on the PR) and skipped, so a broken branch never holds back production.

**The open-PR list is read before anything is built**, so an API failure stops the run rather than deploying a site with no previews. Only same-repo PRs get one: a fork's branch would run with the repository's secrets in scope, which is the boundary the preview workflow has always kept.

    uv run --no-project scripts/deploy_site.py --dry-run   # build everything, push nothing
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent.resolve()

UMBRELLA = "pr-preview"
"""Where the previews live on the branch: `pr-preview/pr-<n>/` under the production site."""

MARKER = "<!-- site-preview -->"
"""Identifies the one comment per PR that carries its preview link, so a rebuild edits it rather than adding another."""

BOT = {
    "GIT_AUTHOR_NAME": "github-actions[bot]",
    "GIT_AUTHOR_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
    "GIT_COMMITTER_NAME": "github-actions[bot]",
    "GIT_COMMITTER_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
}

Builder = Callable[[Path, str | None], Path]
"""Builds the site from a checkout for the given public URL (`None` for production), and returns the directory it wrote."""


@dataclass(frozen=True)
class PullRequest:
    number: int
    sha: str


def git(*args: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    """Run git and return its stdout, raising on a non-zero exit."""
    full = {**os.environ, **env} if env else None
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True, env=full).stdout.strip()


def repo_slug(repo: Path, remote: str) -> str:
    """`owner/name`, from the CI environment if it says, else from the remote URL."""
    if slug := os.environ.get("GITHUB_REPOSITORY"):
        return slug
    url = git("remote", "get-url", remote, cwd=repo)
    match = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?/?$", url)
    if not match:
        raise RuntimeError(f"can't read an owner/name out of the {remote} URL: {url}")
    return match.group(1)


class GitHub:
    """The few REST calls this needs, on the repository's own token."""

    def __init__(self, slug: str, token: str | None):
        self.base = f"https://api.github.com/repos/{slug}"
        self.headers = {"Accept": "application/vnd.github+json", "User-Agent": "mi-ni-deploy-site"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        headers = {**self.headers, **({"Content-Type": "application/json"} if data else {})}
        request = urllib.request.Request(f"{self.base}{path}", data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def paged(self, path: str) -> list[dict[str, Any]]:
        """Every item of a list endpoint, at 100 a page."""
        joiner = "&" if "?" in path else "?"
        items: list[dict[str, Any]] = []
        for page in range(1, 21):
            batch = self.request("GET", f"{path}{joiner}per_page=100&page={page}")
            items.extend(batch)
            if len(batch) < 100:
                return items
        raise RuntimeError(f"{path} runs past 2000 items, which is not a listing this should be paging")


def previewable(pulls: list[dict[str, Any]], slug: str) -> list[PullRequest]:
    """The open PRs that get a preview: those whose head is a branch of this repository."""
    return [
        PullRequest(pull["number"], pull["head"]["sha"])
        for pull in sorted(pulls, key=lambda pull: pull["number"])
        if (pull["head"].get("repo") or {}).get("full_name") == slug
    ]


def checkout(repo: Path, remote: str, ref: str, dest: Path) -> str:
    """A detached worktree at the remote's `ref`, fetched at depth 1 since a build needs the tree and nothing behind it. Returns the sha."""
    git("fetch", "--quiet", "--depth=1", remote, ref, cwd=repo)
    sha = git("rev-parse", "FETCH_HEAD", cwd=repo)
    git("worktree", "add", "--quiet", "--detach", str(dest), sha, cwd=repo)
    return sha


def build_with_uv(worktree: Path, site_url: str | None) -> Path:
    """The real builder: the checkout's own `./go site`, in an environment synced from its own lockfile.

    `--locked` because this doesn't go through `install.sh`, which turns it on under `$CI`: a bare sync would rewrite `uv.lock` to match a `pyproject.toml` that outgrew it, and build against a resolution nobody reviewed. `MINI_SITE_URL` is what keeps a preview's inter-report links and its "← Index" banner inside the preview rather than jumping to production; production leaves it unset and `build_site` derives the URL from the repository.
    """
    env = {key: value for key, value in os.environ.items() if key != "MINI_SITE_URL"}
    if site_url:
        env["MINI_SITE_URL"] = site_url
    subprocess.run(["uv", "sync", "--locked", "--group", "pages"], cwd=worktree, check=True, env=env)
    subprocess.run(["./go", "site"], cwd=worktree, check=True, env=env)
    return worktree / "_site"


def build_into(repo: Path, remote: str, ref: str, worktree: Path, dest: Path, builder: Builder, url: str | None) -> str:
    """Check `ref` out, build it, and move the result to `dest`. Returns the sha that was built. The worktree is removed either way."""
    sha = checkout(repo, remote, ref, worktree)
    try:
        built = builder(worktree, url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(built), str(dest))
    finally:
        git("worktree", "remove", "--force", str(worktree), cwd=repo)
    return sha


def commit_site(repo: Path, site: Path, message: str) -> str:
    """One parentless commit whose tree is `site`, written through a temporary index so the caller's checkout keeps its own.

    `--force` because the exclude rules of the repository this runs in (its `.git/info/exclude`, the global excludes) would otherwise apply to a tree that has nothing to do with them.
    """
    git_dir = Path(git("rev-parse", "--absolute-git-dir", cwd=repo))
    index = git_dir / "deploy-site.index"
    index.unlink(missing_ok=True)
    env = {"GIT_DIR": str(git_dir), "GIT_WORK_TREE": str(site), "GIT_INDEX_FILE": str(index)}
    try:
        git("add", "--all", "--force", ".", cwd=site, env=env)
        tree = git("write-tree", cwd=site, env=env)
    finally:
        index.unlink(missing_ok=True)
    return git("commit-tree", tree, "-m", message, cwd=repo, env=BOT)


def deploy(repo: Path, remote: str, branch: str, site: Path, message: str) -> str | None:
    """Make `site` the branch's only commit. Returns the new tip, or `None` when the branch already serves this tree.

    A force push without a lease, on purpose: the workflow's concurrency group makes this the branch's only writer, and a run that was cancelled mid-push either landed or didn't — the next run rebuilds from state that is at least as fresh and overwrites it either way. Comparing trees first means a run whose event changed nothing (a push to a PR whose reports didn't move, a scheduled pass) pushes nothing, and so triggers no Pages deployment.
    """
    commit = commit_site(repo, site, message)
    tree = git("rev-parse", f"{commit}^{{tree}}", cwd=repo)
    served = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", remote, branch], cwd=repo, capture_output=True, text=True
    )
    if served.returncode == 0:
        git("fetch", "--quiet", "--depth=1", remote, branch, cwd=repo)
        if git("rev-parse", "FETCH_HEAD^{tree}", cwd=repo) == tree:
            return None
    elif served.returncode != 2:  # 2 is "no such ref"; anything else is a fetch problem the push would hit too
        raise subprocess.CalledProcessError(served.returncode, served.args, served.stdout, served.stderr)
    git("push", "--quiet", "--force", remote, f"{commit}:refs/heads/{branch}", cwd=repo)
    return commit


def preview_comment(url: str, sha: str) -> str:
    return (
        f"{MARKER}\n"
        f"**Preview:** {url}\n\n"
        f"Built from {sha[:7]}; ready once the [Pages deployment](../deployments) finishes. "
        "Rebuilt on every push here, and gone once the PR closes."
    )


def failure_comment(sha: str) -> str:
    run = (
        f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        if "GITHUB_RUN_ID" in os.environ
        else None
    )
    where = f" — see [the run]({run})" if run else ""
    return f"{MARKER}\nThe preview for {sha[:7]} failed to build{where}. Production and the other previews deployed without it."


def upsert_comment(api: GitHub, number: int, body: str) -> None:
    """Edit the PR's preview comment in place, or post it the first time. Nothing happens when it already reads `body`."""
    existing = next((c for c in api.paged(f"/issues/{number}/comments") if MARKER in (c.get("body") or "")), None)
    if existing and existing["body"] == body:
        return
    if existing:
        api.request("PATCH", f"/issues/comments/{existing['id']}", {"body": body})
    else:
        api.request("POST", f"/issues/{number}/comments", {"body": body})


def reconcile(
    repo: Path = ROOT,
    remote: str = "origin",
    branch: str = "gh-pages",
    *,
    slug: str | None = None,
    site_url: str | None = None,
    builder: Builder = build_with_uv,
    api: GitHub | None = None,
    dry_run: bool = False,
) -> int:
    """Build production and every open PR's preview, and deploy the lot. Returns the process exit code."""
    slug = slug or repo_slug(repo, remote)
    owner, name = slug.split("/", 1)
    site_url = site_url or f"https://{owner}.github.io/{name}/"
    api = api or GitHub(slug, os.environ.get("GITHUB_TOKEN"))
    pulls = previewable(api.paged("/pulls?state=open"), slug)

    workspace = Path(tempfile.mkdtemp(prefix="deploy-site-"))
    site = workspace / "site"
    built: dict[int, str] = {}
    failed: dict[int, str] = {}
    try:
        main = build_into(repo, remote, "refs/heads/main", workspace / "main", site, builder, None)
        print(f"main@{main[:7]}: built")
        for pull in pulls:
            key = f"pr-{pull.number}"
            try:
                sha = build_into(
                    repo,
                    remote,
                    f"refs/pull/{pull.number}/head",
                    workspace / key,
                    site / UMBRELLA / key,
                    builder,
                    f"{site_url}{UMBRELLA}/{key}/",
                )
                built[pull.number] = sha
                print(f"#{pull.number}@{sha[:7]}: built")
            except Exception as error:  # a broken branch is that PR's problem, and never holds back production
                failed[pull.number] = pull.sha
                print(f"::warning::the preview for #{pull.number} failed to build: {error}")

        listed = " ".join(f"#{n}" for n in built)
        message = f"Deploy site from main@{main[:7]}" + (f" with previews {listed}" if built else "")
        if dry_run:
            print(f"Dry run — nothing pushed. Would deploy: {message}")
            return 0
        tip = deploy(repo, remote, branch, site, message)
        print(f"{branch}: {'unchanged' if tip is None else f'now {tip[:8]}'} — {message}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        git("worktree", "prune", cwd=repo)

    for number, sha in built.items():
        upsert_comment(api, number, preview_comment(f"{site_url}{UMBRELLA}/pr-{number}/", sha))
    for number, sha in failed.items():
        upsert_comment(api, number, failure_comment(sha))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as out:
            out.write(f"{message}\n" + "".join(f"\n- ⚠️ #{n}: preview failed to build" for n in failed))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    parser.add_argument("--repo", type=Path, default=ROOT, help="repository to run from (default: this checkout)")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="gh-pages")
    parser.add_argument("--site-url", help="where the site is served (default: <owner>.github.io/<name>/)")
    parser.add_argument("--dry-run", action="store_true", help="build everything and push nothing")
    args = parser.parse_args()
    return reconcile(args.repo, args.remote, args.branch, site_url=args.site_url, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

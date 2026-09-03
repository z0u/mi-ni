"""The site deploy: does one run leave `gh-pages` serving main plus a preview per open PR and nothing else, as a single commit, and does a broken preview or a repeat run cost anything?"""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import load_script

deploy_site = load_script("deploy_site")

SLUG = "z0u/mi-ni"


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def commit(work: Path, message: str) -> None:
    git("add", "-A", cwd=work)
    git("-c", "user.name=dev", "-c", "user.email=dev@example.invalid", "commit", "-q", "-m", message, cwd=work)


def build(worktree: Path, site_url: str | None) -> Path:
    """A stand-in for `./go site`: names the checkout it ran in and the URL it was given, the two things the real one varies on."""
    who = (worktree / "WHO").read_text()
    if who == "pr-34":
        raise RuntimeError("this branch doesn't build")
    site = worktree / "_site"
    site.mkdir()
    (site / "index.html").write_text(f"<h1>{who}</h1><a href='{site_url or 'https://z0u.github.io/mi-ni/'}'>index</a>")
    (site / ".nojekyll").write_text("")
    return site


class FakeGitHub:
    """Two open PRs, one of them from a fork, and a record of every write."""

    def __init__(self, comments: dict[int, list[dict[str, Any]]] | None = None):
        self.comments = comments or {}
        self.writes: list[tuple[str, str, dict[str, Any] | None]] = []

    def paged(self, path: str) -> list[dict[str, Any]]:
        if path.startswith("/pulls"):
            return [
                {"number": 34, "head": {"sha": "b" * 40, "repo": {"full_name": SLUG}}},
                {"number": 12, "head": {"sha": "a" * 40, "repo": {"full_name": SLUG}}},
                {"number": 56, "head": {"sha": "c" * 40, "repo": {"full_name": "someone/mi-ni"}}},
            ]
        number = int(path.split("/")[2])
        return self.comments.get(number, [])

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        self.writes.append((method, path, body))
        return {}


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """A bare origin: `main`, the head refs of PRs 12 and 34, and a `gh-pages` three deploys deep that still serves a preview for a PR long closed."""
    bare, work = tmp_path / "origin.git", tmp_path / "work"
    git("init", "--bare", "--initial-branch=main", str(bare), cwd=tmp_path)
    git("init", "-q", "--initial-branch=main", str(work), cwd=tmp_path)
    git("remote", "add", "origin", str(bare), cwd=work)
    (work / "WHO").write_text("main")
    commit(work, "source")
    git("push", "-q", "origin", "main", cwd=work)
    for n in (12, 34):
        (work / "WHO").write_text(f"pr-{n}")
        commit(work, f"pr {n}")
        git("push", "-q", "origin", f"HEAD:refs/pull/{n}/head", cwd=work)
        git("reset", "-q", "--hard", "origin/main", cwd=work)

    git("checkout", "-q", "--orphan", "gh-pages", cwd=work)
    git("rm", "-q", "-rf", ".", cwd=work)
    for n in range(3):
        (work / "index.html").write_text(f"<h1>old build {n}</h1>")
        (work / "pr-preview" / "pr-7").mkdir(parents=True, exist_ok=True)
        (work / "pr-preview" / "pr-7" / "index.html").write_text("<h1>a preview whose PR closed weeks ago</h1>")
        commit(work, f"Deploy {n}")
    git("push", "-q", "origin", "gh-pages", cwd=work)
    return bare


@pytest.fixture
def clone(tmp_path: Path, remote: Path) -> Path:
    """The runner as the workflow leaves it: `actions/checkout` clones the triggering ref at depth 1, and nothing has fetched `gh-pages`."""
    path = tmp_path / "runner"
    git("clone", "-q", "--depth", "1", f"file://{remote}", str(path), cwd=tmp_path)
    return path


def served(remote: Path) -> dict[str, str]:
    """Every file on `gh-pages`, path → content."""
    paths = git("ls-tree", "-r", "--name-only", "gh-pages", cwd=remote).splitlines()
    return {path: git("show", f"gh-pages:{path}", cwd=remote) for path in paths}


def test_the_branch_becomes_main_plus_a_preview_per_open_pr(clone: Path, remote: Path):
    api = FakeGitHub()
    assert deploy_site.reconcile(clone, slug=SLUG, builder=build, api=api) == 0

    site = served(remote)
    assert site["index.html"].startswith("<h1>main</h1>"), "production isn't built from main"
    assert "<h1>pr-12</h1>" in site["pr-preview/pr-12/index.html"]
    assert "https://z0u.github.io/mi-ni/pr-preview/pr-12/" in site["pr-preview/pr-12/index.html"], (
        "the preview's links point outside the preview"
    )
    assert ".nojekyll" in site
    assert not [path for path in site if path.startswith("pr-preview/pr-7/")], "a closed PR's preview survived"
    assert not [path for path in site if path.startswith("pr-preview/pr-56/")], "a fork got a preview"
    assert git("rev-list", "--count", "gh-pages", cwd=remote) == "1", "the deploy kept history behind it"


def test_a_broken_preview_is_reported_and_skipped(clone: Path, remote: Path):
    """PR 34's branch doesn't build. Production and the other preview deploy anyway, and its comment says what happened."""
    api = FakeGitHub()
    assert deploy_site.reconcile(clone, slug=SLUG, builder=build, api=api) == 0

    site = served(remote)
    assert "pr-preview/pr-12/index.html" in site
    assert not [path for path in site if path.startswith("pr-preview/pr-34/")]
    posted = {path: body or {} for method, path, body in api.writes if method == "POST"}
    assert "Preview:" in posted["/issues/12/comments"]["body"]
    assert "failed to build" in posted["/issues/34/comments"]["body"]


def test_a_repeat_run_pushes_nothing(clone: Path, remote: Path):
    """The same state builds to the same tree, so the second run leaves the tip alone — and so triggers no Pages deployment."""
    deploy_site.reconcile(clone, slug=SLUG, builder=build, api=FakeGitHub())
    tip = git("rev-parse", "gh-pages", cwd=remote)
    api = FakeGitHub(
        comments={
            12: [
                {
                    "id": 1,
                    "body": deploy_site.preview_comment(
                        "https://z0u.github.io/mi-ni/pr-preview/pr-12/",
                        git("rev-parse", "refs/pull/12/head", cwd=remote),
                    ),
                }
            ]
        }
    )
    assert deploy_site.reconcile(clone, slug=SLUG, builder=build, api=api) == 0
    assert git("rev-parse", "gh-pages", cwd=remote) == tip
    assert not [w for w in api.writes if w[1] == "/issues/12/comments"], "an unchanged preview rewrote its comment"


def test_a_dry_run_pushes_nothing(clone: Path, remote: Path):
    before = git("rev-parse", "gh-pages", cwd=remote)
    api = FakeGitHub()
    assert deploy_site.reconcile(clone, slug=SLUG, builder=build, api=api, dry_run=True) == 0
    assert git("rev-parse", "gh-pages", cwd=remote) == before
    assert api.writes == []


def test_the_runners_checkout_is_left_alone(clone: Path):
    """Builds happen in worktrees that are removed afterwards, and the commit goes through a temporary index."""
    deploy_site.reconcile(clone, slug=SLUG, builder=build, api=FakeGitHub())
    assert git("status", "--porcelain", cwd=clone) == ""
    assert (clone / "WHO").read_text() == "main"
    assert len(git("worktree", "list", cwd=clone).splitlines()) == 1
    assert not (clone / ".git" / "deploy-site.index").exists()


def test_previewable_keeps_same_repo_heads_in_number_order():
    pulls = FakeGitHub().paged("/pulls?state=open")
    assert deploy_site.previewable(pulls, SLUG) == [
        deploy_site.PullRequest(12, "a" * 40),
        deploy_site.PullRequest(34, "b" * 40),
    ]
    assert deploy_site.previewable([{"number": 1, "head": {"sha": "d" * 40, "repo": None}}], SLUG) == [], (
        "a PR whose fork was deleted has no head to build"
    )


def test_the_preview_comment_is_edited_in_place():
    api = FakeGitHub(
        comments={
            12: [
                {"id": 7, "body": "an unrelated comment"},
                {"id": 8, "body": f"{deploy_site.MARKER}\nan older link"},
            ]
        }
    )
    deploy_site.upsert_comment(api, 12, "new body")
    assert api.writes == [("PATCH", "/issues/comments/8", {"body": "new body"})]

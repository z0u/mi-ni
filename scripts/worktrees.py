#!/usr/bin/env python
"""Agent worktrees, and which of them are finished with.

Claude Code gives each agent its own git worktree under `.claude/worktrees/`, inside the repo rather than beside it. The location is the harness's choice and there is no setting exposed to move it, so the checkouts land in the tree whether or not anything is watching. Nothing removes one when its branch is done: the oldest found so far had sat there for ten days after its PR merged, clean and level with `main` the whole time, and only surfaced because it was breaking pytest collection. That collection noise is fixed, which makes the next stranded worktree quieter rather than rarer — hence this.

`--prune` removes the ones that have plainly finished, and only those. Two conditions, both required:

- **Clean.** `git status --porcelain` is empty. Untracked files count, because half-finished work nobody committed is exactly what a session that ended badly leaves behind, and it is not ours to throw away. Ignored files do not count, and that is the deliberate limit of the protection: `__pycache__/` appears the moment anything in the worktree runs, so counting ignored files would make every worktree unprunable within a minute of being used. `.gitignore` is the repo saying which files it is willing to lose, and this takes it at its word — including `scratch*` and `.mini/`, which is where a worktree's local store cache lands.
- **Landed.** Either the branch head is an ancestor of the integration ref, or its tree is byte-identical to it. The second test is what covers a squash merge, where the branch keeps commits that GitHub collapsed into one and an ancestor test alone would hold the worktree forever.

Anything failing either test is kept, listed with the reason, and given the command to remove it by hand. That asymmetry is deliberate: a routine calling this on a schedule should be able to delete only what it can show is redundant, and should say so out loud when it declines rather than deciding on its own.

Nothing here touches the network, so `origin/main` is read at whatever it was last fetched to. A stale ref makes fewer things look landed, never more, which is the safe direction to be wrong in.

Branches are left alone after their worktree goes. `git worktree remove` does not delete one, and neither do we — a branch is a few bytes and a name someone may still want, while the checkout it was mounted in is tens of megabytes and reproducible from the branch alone. That is also what makes the tree-identity test safe to rely on: a branch can reach it by a route other than a squash merge, having arrived at the same content by its own commits, and removing the checkout still costs nothing, because the ref survives and `git worktree add` rebuilds the rest.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
NEST = Path(".claude") / "worktrees"  # where the harness puts them, relative to the repo
INTEGRATION = ("origin/main", "main")  # first one that resolves wins


def git(*args: str, cwd: Path = ROOT) -> str:
    """Run git and return its stdout, raising on a non-zero exit."""
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def git_ok(*args: str, cwd: Path = ROOT) -> bool:
    """Run git for its exit code alone — the shape `--is-ancestor` and `diff --quiet` are built for."""
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).returncode == 0


@dataclass(frozen=True, slots=True)
class Worktree:
    """One entry from `git worktree list --porcelain`."""

    path: Path
    head: str
    branch: str | None
    locked: bool
    root: Path = ROOT

    @property
    def nested(self) -> bool:
        """Whether this is an agent worktree, i.e. one inside the repo under `.claude/worktrees/`.

        Only nested worktrees are ever removed. A sibling checkout beside the repo is someone's deliberate workspace, and the problem this script exists for — a checkout that the tree-walking tools find a second copy of the project in — is only the nested kind.
        """
        return self.path.is_relative_to(self.root / NEST)

    @property
    def label(self) -> str:
        try:
            return str(self.path.relative_to(self.root))
        except ValueError:
            return str(self.path)


@dataclass(frozen=True, slots=True)
class Status:
    """A worktree judged against the integration ref."""

    tree: Worktree
    dirty: bool
    landed: bool

    @property
    def removable(self) -> bool:
        return self.tree.nested and not self.tree.locked and not self.dirty and self.landed

    def why(self, integration: str) -> str:
        """One line saying what was decided, and for anything kept, what would have to change."""
        if not self.tree.nested:
            return "outside the repo — left alone"
        if self.tree.locked:
            return "locked"
        if self.dirty:
            return "uncommitted or untracked changes"
        if not self.landed:
            return f"commits not in {integration}"
        return f"landed in {integration}, clean"


def parse(porcelain: str, root: Path = ROOT) -> list[Worktree]:
    """Split `git worktree list --porcelain` into records, dropping the main worktree.

    Records are blank-line separated, one `key value` per line, with `detached` and `locked` appearing as bare flags. The main worktree is the repo itself, so it goes: it is never a candidate for removal, and listing it would only invite the question.

    Records whose directory no longer exists go too. `git worktree prune` clears most of those before we look, but it leaves a locked one behind, and there is nothing useful to say about a checkout that isn't there — while judging it would mean running `git status` in a missing directory.
    """
    trees = []
    for block in porcelain.split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.strip().splitlines():
            key, _, value = line.partition(" ")
            fields[key] = value
        if "worktree" not in fields:
            continue
        trees.append(
            Worktree(
                path=Path(fields["worktree"]).resolve(),
                head=fields.get("HEAD", ""),
                branch=fields.get("branch", "").removeprefix("refs/heads/") or None,
                locked="locked" in fields,
                root=root,
            )
        )
    return [t for t in trees if t.path != root and t.path.is_dir()]


def integration_ref(cwd: Path = ROOT) -> str | None:
    """The ref a finished branch should have landed in, or `None` if neither candidate resolves."""
    return next((ref for ref in INTEGRATION if git_ok("rev-parse", "--verify", "--quiet", ref, cwd=cwd)), None)


def judge(tree: Worktree, integration: str) -> Status:
    """Whether *tree* has uncommitted work, and whether its content reached *integration*.

    Both git calls run inside the worktree, which shares its object database with the repo, so the refs resolve the same either way and this needs no separate handle on the root.

    The `diff --quiet` half does work the ancestor test can't. A squash-merged branch keeps the commits that GitHub collapsed into one, so it is never an ancestor of `main`; what it does have, once it is level, is an identical tree. Either signal alone means the checkout holds nothing that would be lost.
    """
    dirty = bool(git("status", "--porcelain", cwd=tree.path))
    landed = git_ok("merge-base", "--is-ancestor", tree.head, integration, cwd=tree.path) or git_ok(
        "diff", "--quiet", integration, tree.head, cwd=tree.path
    )
    return Status(tree=tree, dirty=dirty, landed=landed)


def render(statuses: list[Status], integration: str) -> str:
    """The worktrees as an aligned listing, removable ones first."""
    if not statuses:
        return "No worktrees besides the repo itself."
    ordered = sorted(statuses, key=lambda s: (not s.removable, s.tree.label))
    width = max(len(s.tree.label) for s in ordered)
    branch_width = max(len(s.tree.branch or "(detached)") for s in ordered)
    return "\n".join(
        f"  {'✂' if s.removable else ' '} {s.tree.label:<{width}}  {s.tree.branch or '(detached)':<{branch_width}}  {s.why(integration)}"
        for s in ordered
    )


def remove(status: Status, dry_run: bool) -> bool:
    """Remove one worktree, reporting what happened. Returns whether it went, which sets the exit code.

    A removal can still fail after the judging says it shouldn't — a file held open, a permission — and a routine pruning on a schedule should hear about that rather than read the summary and assume.
    """
    if dry_run:
        print(f"  would remove {status.tree.label}")
        return True
    try:
        git("worktree", "remove", str(status.tree.path))
    except subprocess.CalledProcessError as e:
        print(f"  ❌ {status.tree.label}: {e.stderr.strip()}", file=sys.stderr)
        return False
    print(f"  ✂ removed {status.tree.label}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prune", action="store_true", help="remove agent worktrees that are clean and landed")
    ap.add_argument("--dry-run", action="store_true", help="with --prune, say what would go without touching anything")
    args = ap.parse_args()

    # Clears git's own metadata for worktrees whose directory was deleted by hand. Always safe:
    # it only forgets administrative entries that no longer point at a checkout.
    git("worktree", "prune")

    if not (integration := integration_ref()):
        print(f"Can't judge what has landed: none of {', '.join(INTEGRATION)} resolve.", file=sys.stderr)
        sys.exit(1)

    statuses = [judge(t, integration) for t in parse(git("worktree", "list", "--porcelain"))]
    removable = [s for s in statuses if s.removable]

    if not args.prune:
        print(render(statuses, integration))
        if removable:
            print(f"\n{len(removable)} finished — remove with `./go worktrees --prune`.")
        return

    failed = [s for s in removable if not remove(s, args.dry_run)]
    if kept := [s for s in statuses if not s.removable and s.tree.nested]:
        print(f"\nKept {len(kept)}, needing a look first:")
        for status in kept:
            print(f"  {status.tree.label} — {status.why(integration)}")
            print(f"    git worktree remove --force {status.tree.label}")
    if not removable and not kept:
        print("Nothing to prune.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

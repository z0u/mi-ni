---
status: done
tags: [agents, testing, tooling]
opened: 2026-08-14
closed: 2026-08-15
---
# Agent worktrees live inside the repo, and nothing clears them out

Agents get their own git worktree under `.claude/worktrees/`, which is inside the repo rather than beside it. That placement has two consequences: tools that walk the tree find a whole second checkout, and nothing removes the checkout once its branch is finished.

**Collection, now fixed.** `pytest` walked into `.claude/worktrees/` and collected a second copy of `tests/`, reporting four `ImportPathMismatchError`s on every `./go check`. The cause is that setting `norecursedirs` in `pyproject.toml` *replaces* pytest's default list rather than extending it, and `.*` is the default entry that keeps it out of dotted directories — so listing `node_modules` and `.venv` quietly opened `.claude` up. Fixed by putting `.*` back at the head of the list. The general lesson is worth carrying to any other tool configured with a directory-exclusion list: check whether the setting extends the defaults or replaces them. `ruff`, `ty`, `vulture`, and the `marimo check` hook each have their own, and none of them have been looked at for this.

**The other exclusion lists, checked 2026-08-15.** Only pytest was ever exposed. `ruff` and `ty` stay out because both respect `.gitignore` by default and `.claude/worktrees/` is listed there — not because of anything in their own excludes, which is worth knowing, since `ruff --no-respect-gitignore` picks up all 190 files in a scratch worktree. `vulture` stays out because `[tool.vulture] paths` names four explicit roots, so it is never offered the directory; its `exclude` list replaces the defaults exactly as pytest's does, and lists only build detritus. `marimo check` is only ever invoked on named files, by the `PostToolUse` hook and by lint-staged, so it has no tree to walk. Three different mechanisms, none of them the exclusion list, and each one quiet enough to be removed by accident — so `tests/test_worktrees.py` guards all three, with a comment saying which tool each protects.

**Placement.** The worktrees do have to be inside the repo. The path is the harness's, and there is no setting for it: nothing in `.claude/settings.json`, no `CLAUDE_*` environment variable, and no key in the settings schema that the Claude Code bundle reads. So the placement half can't be designed away, and the lifecycle half stands on its own.

**Lifecycle, now fixed.** `./go worktrees` lists agent worktrees with a verdict each, and `./go worktrees --prune` removes the ones that are clean and have landed. Landed means the branch head is an ancestor of `origin/main`, *or* its tree is identical to it — the second test is what covers a squash merge, where the branch keeps the commits GitHub collapsed into one and an ancestor test alone would hold the checkout forever. It removes nothing with uncommitted or untracked work, nothing locked, and nothing outside the repo; anything held back is listed with the reason and the command to remove it by hand. Branches survive their worktrees, which is also what makes the tree-identity test cheap to be wrong about: the checkout is reproducible from the ref.

The three options this item weighed went one-and-a-half ways. The prune command is the substance, and `AGENTS.md` gained a line pointing at it rather than asking a session to tidy up after itself — a stranded worktree is evidence that a session didn't end tidily, so that instruction would fire in exactly the cases it isn't needed. The `./go check` warning was left out: it would put a git walk on the hot path to say something `./go worktrees` says on demand.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/worktrees-live-inside-the-repo.md`](https://github.com/z0u/sca2/blob/main/todo/eng/worktrees-live-inside-the-repo.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on. `tests/test_worktrees.py` and `./go worktrees` are here; the `AGENTS.md` line pointing at the prune is not, which is one sentence to add.

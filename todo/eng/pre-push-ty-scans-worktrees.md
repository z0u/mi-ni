---
status: done
tags: [tooling]
opened: 2026-07-29
closed: 2026-08-09
---
# Pre-push `ty` check scanning nested worktrees

Filed against ty 0.0.49: pushing from a `.claude/worktrees/<name>` worktree ran `ty check` from the parent checkout, which picked up the worktree's copy of `src/mini` as a separate tree with the wrong first-party roots — 23 spurious unresolved-import/type errors on a Markdown-only change. Re-tested on current pin (ty 0.0.65, bumped 2026-07-30 in the entry just above): built a scratch worktree under `.claude/worktrees/` and ran the hook's actual command, `uv run ty check`, from the repo root exactly as `pre-push-check.sh` does. Clean — "All checks passed!". Forcing `--no-respect-ignore-files` reproduces the original failure verbatim (23 diagnostics, all `unresolved-import` under `.claude/worktrees/tytest/src/mini/...`), confirming the fix is ty now respecting `.gitignore` by default (`.claude/worktrees/` is listed there) — gained somewhere in the 0.0.49 → 0.0.63 bump, not anything in this repo. No code change needed; left as a config default rather than an explicit `[tool.ty]` exclude, since pinning to the current default would just be encoding today's behavior as if it were a requirement.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/pre-push-ty-scans-worktrees.md`](https://github.com/z0u/sca2/blob/main/todo/eng/pre-push-ty-scans-worktrees.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

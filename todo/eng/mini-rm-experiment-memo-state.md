---
status: open
tags: [cli]
opened: 2026-07-14
bundle: cli-devx
---
# No way to delete an experiment's memo state

From the 2026-07-14 cold-exploration session on CLI usability; the copy-pasteable-hints / sorting / help-text tier shipped (see [mi-ni#57](https://github.com/z0u/mi-ni/issues/57) for the running thread).

`mini gc <name>` sweeps only stale attempt files/superseded records, so a scratch or renamed experiment's DONE records live forever — on Modal too (a `cli-probe` probe experiment now sits there as a permanent example). Wants a `mini rm <name>` with the same dry-run-by-default posture as gc. The manual escape hatch, verified: `bin/modal dict delete mini-cp-<name> --yes` plus `bin/modal volume delete <name> --yes` clears both planes — which is roughly what `mini rm` would wrap.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/mini-rm-experiment-memo-state.md`](https://github.com/z0u/sca2/blob/main/todo/eng/mini-rm-experiment-memo-state.md) there) with the code it describes; the stranded `cli-probe` experiment is on that project's Modal workspace.

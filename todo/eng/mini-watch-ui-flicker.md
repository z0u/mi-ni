---
status: open
tags: [cli]
opened: 2026-09-01
bundle: cli-devx
---
# Status redraw causes flicker in terminal

When running `mini watch ...`, it draws one row per task, each with a progress bar. The first bar updates OK, but the others all flicker when they are redrawn.

```
❯ mini watch ex-2.2.1
score_run-0d869395648e                     ━━━━━━━━ 100% 0:00:00
score_run-1b4eea3e6d55  !! IndexError: ... ━━━━━━━━   0% 0:00:00
score_run-3fd612c1f6d3                     ━━━━━━━━ 100% 0:00:00
```

Perhaps it needs to be batched into a single update or [un]buffered.

Observed in VS Code. Unsure if it happens in other terminal emulators.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/mini-watch-ui-flicker.md`](https://github.com/z0u/sca2/blob/main/todo/eng/mini-watch-ui-flicker.md) there) with the code it describes; the transcript is from that project's `ex-2.2.1`.

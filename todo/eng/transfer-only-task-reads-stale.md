---
status: done
tags: [cli]
opened: 2026-08-11
closed: 2026-08-12
---
# A task whose whole body is store transfers read as `⚠ stale — worker may be dead`

`publish_ablations` downloads 27 array artifacts and uploads one, so it emits no step progress for its entire run; `status` badged it stale at 300s+ and `watch` exited 3 (attention) three times while it was healthy. Heartbeats ride on progress emissions, so with no emissions the heartbeat sat at `started_at` — and `stale_heartbeat` never read `phase_until`, which only `stale_progress` consulted. Both badges now pause inside a declared span, through a shared `runs.in_declared_phase`.

The spans turned out to be the easy half. Sampling the real shape (a scratch repro driving `put`/`get` under a `_phase_hook`) showed the false alarm lives in the *gaps between* transfers, where the record carries no deadline at all and the step has not moved since `started_at` — so `stale_progress` still cried wolf on every poll landing between two artifacts. Faking `progress_at` there was the wrong fix: a frozen step alongside a live heartbeat is the genuine wedge signature, and blurring it would blind the detector for any training loop that checkpoints mid-loop. Instead the worker stamps `phase_at` at each span boundary, and `stale_progress` reads a recently-crossed boundary as movement — not step-shaped, but movement. It buys exactly one threshold, the same as a step emission would, so a task that closes a span and *then* wedges is still caught (verified). Phase transitions also beat the heartbeat now, since writing to the control plane is the worker breathing.

Display side, the follow-on noted under the watchdog entry below: a live span is named on the human-readable line (`⋯ put model`) instead of leaving `3300/3300` sitting there, which is the shape a wedge has too. `phase_at` joins `phase`/`phase_until` in `status --json`. Regression tests cover the composition, not the pieces alone — the gap between spans is exactly what unit tests on either half would have missed.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/transfer-only-task-reads-stale.md`](https://github.com/z0u/sca2/blob/main/todo/eng/transfer-only-task-reads-stale.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on; `publish_ablations` is a task in that project.

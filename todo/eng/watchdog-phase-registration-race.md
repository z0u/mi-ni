---
status: done
tags: [ci, mini, tests]
opened: 2026-09-09
closed: 2026-09-10
---
# `blocking_phase` can lose a race with the watchdog on a slow runner

`tests/mini/test_watchdog.py::test_blocking_phase_lets_a_post_loop_upload_finish` failed once on CI ([run 34319699866](https://github.com/z0u/sca2/actions/runs/34319699866), on `5a3233b`) and passes locally three times in a row. The record's error was:

```
WatchdogStall: no step progress in 0s (watchdog 0.25s) at step 3/3
```

Two things in that line: the stall is under a second (`:.0f` rounds it to `0s`), so the watchdog polled within one or two poll intervals of the last step; and the named limit is the 0.25 s step watchdog, not the 30 s phase the task had declared. So at the moment the watchdog checked, the phase was not yet visible to it. The task enters `blocking_phase` right after its last `emit_progress`, so on a loaded runner the gap between the emission and the phase's registration can outlast a 0.25 s watchdog whose poll is `timeout_s / 4`.

Worth a look at how a phase reaches the watchdog (`src/mini/_watchdog.py`, `_widest_phase`, and `mini.progress.blocking_phase`): if registration goes through the same channel as progress and is ordered after the emission, the fix might be as small as having the phase's `__enter__` register before it returns, or having the test's watchdog poll less tightly. A watchdog this tight is a test setting; production watchdogs are seconds to minutes, so a real run would need a very long scheduling gap to hit this.

## Notes

**2026-09-10, tech debt** — fixed in `_phase_hook` (`src/mini/_taskworker.py`), and it was a production bug rather than only a test artifact. Registration is synchronous, as hoped, but it was sequenced *after* the hook's own control-plane write: `phase()` registered the span in `open_spans`, called `restamp()` — `record()` → `store.update_if`, which locally takes a store-wide `flock` and does a JSON read-modify-write, and on Modal is a network round trip — and only then entered `watchdog.phase`. That write makes no step progress either, so it ran under the tight step threshold, which is why the diagnosis named the 0.25 s step watchdog rather than the 30 s phase the task had just declared. The exit path had the mirror of it: the closing stamp ran after `watchdog.phase` had already closed. Under a `ctx.map` the wait is unbounded, since the stamp can queue behind any sibling worker's heartbeat.

The fix makes `watchdog.phase` the outermost thing in the hook, so the span's budget covers both stamps as well as the body between them. `test_a_phase_covers_the_stamps_that_open_and_close_it` pins the ordering deterministically: it drives `_phase_hook` with a slow `record` and asserts the threshold in force during each stamp is the span's, not the step one — `[0.2, 0.2]` against the old ordering, `[30.0, 30.0]` now. The residual window is what remains between the last `emit_progress` and the phase's `__enter__`, now a handful of instructions rather than a locked file write, so the test's 0.25 s watchdog was left as it stood; loosening it would have cost the test its point.

One relative left unfixed, noted rather than acted on: `Store.put` computes `_size_on_disk(src)` before opening its phase, because the budget is sized from the answer. That is a local stat walk rather than a lock or a round trip, so it is a much smaller exposure — but a checkpoint tree with very many files under a tight watchdog is the shape that would find it.

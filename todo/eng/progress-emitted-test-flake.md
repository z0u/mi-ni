---
status: done
tags: [testing]
closed: 2026-08-13
---
# `test_progress_emitted_during_training` flaked once under load

**Fixed by [sca2#78](https://github.com/z0u/sca2/pull/78)**, which turned out to already cover this: the training loop's placeholder `total=0` message races the real `emit_progress` total, and the old set-equality assertion held the placeholder to the same standard as a real total. [sca2#78](https://github.com/z0u/sca2/pull/78) filters it out (`{m.total for m in messages if m.total} == {max(steps)}`) and adds a check on the final message's step/total instead. This item was migrated from a second, differently-worded note describing the same assertion and the same root cause — it just never got removed when [sca2#78](https://github.com/z0u/sca2/pull/78) landed. Original note follows. `tests/experiment/test_training.py::test_progress_emitted_during_training` failed once under `-n auto` on a loaded container ([sca2#73](https://github.com/z0u/sca2/pull/73)), at `{m.total for m in messages} == {max(steps)}`, and passed on the next three full-suite runs and on a serial re-run. Not reproduced since, so the cause is unconfirmed. `total` is constant across `emit_progress` calls and `BackgroundEmitter.close` flushes synchronously, so the likely shape is a second message reaching the queue from the metrics path with a different `total` — i.e. the assertion is stricter about queue contents than the contract really is. Worth reproducing under artificial load before changing anything. Noted while upgrading dependencies; nothing in that upgrade touches the debouncer, but it has not been ruled out either.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/progress-emitted-test-flake.md`](https://github.com/z0u/sca2/blob/main/todo/eng/progress-emitted-test-flake.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

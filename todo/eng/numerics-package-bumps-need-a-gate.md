---
status: done
tags: [determinism, memoization]
opened: 2026-08-05
closed: 2026-08-19
---
# Numerics-relevant package bumps need a gate

Found while working on [sca2#73](https://github.com/z0u/sca2/pull/73).

A jax 0.10.1 → 0.11.0 upgrade changed the final-weights digest of a fixed nGPT d64-L4 run on a fixed L4 (`e4b7c106…` → `884ee00c…`, loss drift ~1 part in 4×10⁶), while the memo key stayed identical — package versions feed neither `task_key_parts`' identity nor its validity evidence, since `_is_project_file` excludes site-packages. So a DONE record survives the upgrade serving its old number, and a re-run for any other reason quietly writes a new one under the same key. Written up in `eng/determinism.md`. Options, none implemented: fold a pinned set of numerics packages into the evidence (invalidates broadly, and every past record with it); surface a warning when a record's `compute_env` numerics differ from the current environment (cheap, detection-only, probably the right first move); or keep it manual and bump `version=` per affected task on each numerics upgrade.

Closed with the middle option. `compute_env` now records `jax`/`jaxlib`/`numpy` versions on every attempt (`env.numerics_packages`), `mini.runs.numerics_drift` compares a record's set against the installed one, a tick warns once per process when it serves hits that predate an upgrade, and `mini status` re-derives the same thing from the records alone (a `⚠` line, and `numerics_drift` under `--json`). Nothing invalidates: the levers stay `version=` and an honest note beside published numbers. Written up in `eng/determinism.md` under "The gate: detection, not invalidation".

## Notes

**2026-08-19, tech-debt session** — two limits worth knowing, both stated in the doc. The list of watched packages is what a measurement has implicated, so a bump elsewhere (`optax`, `scipy`, `equinox`) is still silent; extending `_NUMERICS_PACKAGES` only affects records written afterwards. And the comparison is against the *driver's* installed set, since a read-only view has nothing else to compare with — fine while the Modal image is frozen from the same lock, misleading if something is ever pinned apart deliberately. Both would want a new item rather than reopening this one.

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/numerics-package-bumps-need-a-gate.md`](https://github.com/z0u/sca2/blob/main/todo/eng/numerics-package-bumps-need-a-gate.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on. [Plain `watch` omits the numerics-drift note](./watch-omits-the-numerics-drift-note.md) is the loose end here.

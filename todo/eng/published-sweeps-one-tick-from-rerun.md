---
status: open
tags: [memoization, storage]
opened: 2026-07-27
---
# Published sweeps are one tick away from a full re-run after an evidence-scheme change

Widening what the fingerprint tracks re-stamps every task's evidence, so the next `mini run` re-runs the whole DAG in place, even though no experiment code moved. Adding a small step to a published sweep tripped this: the deferred-import tracing ([sca2#58](https://github.com/z0u/sca2/pull/58), [sca2#59](https://github.com/z0u/sca2/pull/59)) landed after the sweep, so the tick re-ran the corpus step and would have re-trained all 24 cells. Cost is the smaller half of the problem; the real one is that a re-trained sweep may not reproduce the numbers a published report already quotes (determinism landed after that run too), so the report and the store would silently disagree. Nothing to fix in the mechanism itself — over-invalidation is the right bias — but two things would help. A read-only `mini plan <exp>` that lists what a tick would launch and why, so the choice to re-run is made before the launch and not after; and something that records, per published ref, the evidence the run was produced under, so "this report's numbers predate the current scheme" is a fact the report can state rather than a thing you rediscover. The workaround for now is a standalone script that reads the published checkpoints and writes results back under their own ref, leaving the DAG alone (sca2's `docs/m2/ex-2.1.5/cross_eval.py` is the worked example). [A documentation-only edit in `src/` re-runs the DAG](./docstring-edits-move-the-memo-fingerprint.md) was the neighbouring trigger, since closed.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/published-sweeps-one-tick-from-rerun.md`](https://github.com/z0u/sca2/blob/main/todo/eng/published-sweeps-one-tick-from-rerun.md) there) with the code it describes; the sweep that tripped it was that project's ex-2.1.5.

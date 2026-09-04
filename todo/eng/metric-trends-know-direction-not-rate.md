---
status: open
tags: [cli, monitoring]
opened: 2026-07-27
---
# Metric trends know a direction, not a rate

Raised in the review of PR [sca2#58](https://github.com/z0u/sca2/pull/58).

`expect_metrics`, wrong-way window counting, and a sample floor on the window all shipped; what's left is how coarse the judgment is. A window mean is compared without reference to the within-window spread, so a metric with a genuinely wide spread can still string together three wrong-way windows by chance — if that starts crying wolf, judge the movement against the spread the worker already has the samples to compute (a running sum of squares would do it, alongside the sum it already keeps). And a direction can't catch a loss that is descending far too slowly to reach anything useful inside the budget: that reads as perfectly healthy. A projected-final-value flag would be the counterpart to the timeout projection.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/metric-trends-know-direction-not-rate.md`](https://github.com/z0u/sca2/blob/main/todo/eng/metric-trends-know-direction-not-rate.md) there) with the code it describes.

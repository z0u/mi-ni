---
status: done
tags: [cli]
opened: 2026-08-11
closed: 2026-08-11
---
# `mini results` spent its width on digits instead of on the curve

The elided view kept a sequence's first three elements and a count, so a 100-epoch `val_loss` printed three 17-digit floats and answered nothing you would ask of a metric trace — not where the run ended, not whether it spiked, not what the best epoch reached. A numeric sequence past 8 elements now summarizes over the whole of it instead: first → last, any interior min/max, and mean/std at `.3g`. An extreme is named only where it is an interior one, since a trace usually runs one way and its extremes are then the endpoints already printed — so an absent `min` reads as "never went below where you can see it end", and that rule is where most of the width went. Measured on an ex-2.1.10-shaped `eval_one` result: 13,372 chars raw, 1,257 elided, 1,050 summarized — 16% under the old view while carrying the trend it didn't. Rounding is the one thing in this view that isn't verbatim, and the length floor is what contains it: a sequence short enough to print in full still prints in full and exact, so the trade is only ever made where the alternative was three exact elements out of a hundred. `--full` still has the originals. Passed over rather than deferred: numpy arrays still print shape only. Results here go through `.tolist()`, so sequences cover the real payloads, and a bare `float32[100]: 5.19 → 1.28, …` sitting in a dict rendering reads as though its commas were the dict's.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/mini-results-summarize-sequences.md`](https://github.com/z0u/sca2/blob/main/todo/eng/mini-results-summarize-sequences.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

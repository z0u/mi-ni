---
status: open
tags: [agents, reports, review]
opened: 2026-07-31
---
# Cut the cost of a review round, but guard the right half

One `results-reviewer` pass spent ~97k tokens, and by its own account most of it went on recomputing every quoted number from the store: all seven conditions' margins, ᾱ, EM, NLL; every grading ρ and R²; per-seed retention and peak epochs; margin-by-layer; geometry; the corpus nulls. A second round would redo all of it. Three rounds is not itself the problem — each round found real defects — but the cost per round is mostly spent in the wrong place, and it scales with the report rather than with what changed.

Which half broke: every quoted number was already correct, because reports read them from the store through f-strings at render time, so there is no transcription step to get wrong. All four real defects were natural-language relations wrapped around correct numbers: "well under" a null the value sits above; a span/op1 seam the rendered list contradicts; "far above" a ceiling cleared by 0.05; an effect attributed to the factor with the smaller main effect. Re-deriving numbers catches none of these.

Suggestions (but think carefully before implementing):

- Make comparisons computed rather than asserted. The deep fix, and the same trick that made the numbers safe in the first place. A small helper in the report layer — `rel(a, b, noise=...)` rendering "above" / "below" / "level with", and "clears by {x}" in place of "far above" — turns an inverted comparison from a prose bug into an impossibility. Every one of the four defects above was a relation a helper could have rendered.
- Make comparisons easy to check. As above, but fail loudly if it's not what is expected; an inline test. Probably easiest to put an `assert <condition>` in the cell; for something inline in the prose we could use a matcher library, along the lines of:

  ```py
  mo.md(f"""
  Some long paragraph...
  x is above{expect(x).toBeGreaterThan(floor, noise=...)} the floor
  """)
  ```

  ... but this needs more design and a cost/benefit analysis.

- Single-source the cross-experiment constants. These are the only numbers a reviewer must visit another experiment's store to check, and they are currently pasted and duplicated: one upstream margin appeared in a report four times under three names, plus a `_REFS` table for another experiment. They should live in `experiment.py` once, with provenance (ref, cells, statistic) in the docstring. Then it is one block to verify instead of a hunt, and it is the natural thing for a ledger to key on.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/cut-review-round-cost.md`](https://github.com/z0u/sca2/blob/main/todo/eng/cut-review-round-cost.md) there) with the code it describes; the review round and the constants are that project's ex-2.1.7.

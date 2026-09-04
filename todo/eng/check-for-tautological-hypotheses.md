---
status: open
tags: [agents, methodology, skills]
opened: 2026-07-31
---
# Teach the skills and review agents to check for tautologies

Ex-2.1.7's H4(a) scored the containment of ᾱ in conditions carrying a term that is _defined_ as a penalty on ᾱ, so "the repulsive term lowers ᾱ" was never going to be news: the gate could only measure whether the weight was large enough, not whether the mechanism explains anything. It went unnoticed through preregistration and a prereg review round, and was caught only when writing up the results. The general shape: a hypothesis whose statistic is the quantity a treatment directly optimizes, or a metric that shares its definition with the intervention. Two places to fix it. (1) The `science` skill should name the failure mode when discussing how to choose a scored statistic, with the rule of thumb that a gate is informative only if a reader cannot predict its direction from the method section alone. (2) The `prereg-reviewer` agent should carry an explicit check — for each hypothesis, ask what the treatment optimizes and whether the scored statistic is that quantity, a monotone function of it, or independent of it; only the third is a real test. `results-reviewer` should carry the mirror check for claims built on such a gate. Worth a look at whether earlier reports have the same shape.

Same family, second instance, found the same day. Ex-2.1.7's op1-only factor is not the pure narrowing H3 reads it as: the anchor term normalizes by the _realized_ mask, so pulling one position instead of four divides the same λ among ~3.9× fewer positions and makes each ~3.9× stronger. The factor changes which positions are pulled and how hard, and nothing in the design said so. The reading survived only because the ceiling arm happens to disambiguate it (10× the per-position gradient made selectivity worse, so strength is not the active ingredient) — luck, not design. So the reviewer check should be broader than tautology: for each factor, enumerate everything it changes, not just the thing it is named for, and pay particular attention to normalizers, denominators, and anything averaged over a set whose size the factor alters. A useful prompt: "if I renamed this factor after its side effect instead of its intent, would the hypothesis still read as written?" The design-side fix here is to normalize by a fixed count rather than the realized mask, which the experiment's loss code should offer.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/check-for-tautological-hypotheses.md`](https://github.com/z0u/sca2/blob/main/todo/eng/check-for-tautological-hypotheses.md) there) with the code it describes; both instances are that project's ex-2.1.7, kept here because they are the evidence.

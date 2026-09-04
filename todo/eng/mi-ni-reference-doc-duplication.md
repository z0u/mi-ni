---
status: done
tags: [docs, skills]
opened: 2026-08-02
closed: 2026-08-22
---
# Duplication across the mi-ni reference docs

Found while running the writing/text-lint passes over `.agents/skills/`.

The cache-friendliness guidance (narrow inputs, cheap `main`, folded RNG seeds, forcing a re-run) is near-verbatim in both `authoring.md` and `memoization.md`; the `Experiment(deps=[...])` note is in both `storage.md` and `running.md`; "failure is terminal by design" is in both `recovery.md` and `running.md`; and `memoization.md` points at `recovery.md` twice saying much the same thing. Each doc reads fine alone, so this only costs when they drift. Also: `reports.md`'s `### Simplification pass` is an h3 under the h1 with no h2 above it, and is arguably writing-workflow guidance that belongs with the `writing`/`text-lint` skills rather than in a publishing reference.

## Notes

**2026-08-22, tech debt** — Done. Ownership was decided by what `SKILL.md` already advertises for each doc, which settled every case without fresh judgment. It gives `authoring.md` the cache-friendly design and `memoization.md` the identity/evidence model, so the duplicated habits section left memoization; its `ctx.map` zip note moved up to authoring's opening example, where it now explains the `[vocab] * len(LRS)` idiom that had gone unexplained. `running.md` owns `mini lineage`, so it keeps the upstream capture and `Experiment(deps=[...])`, and `storage.md` links across.

Two duplications beyond the four listed. `running.md`'s `## Recovery` preamble was restating three things at once — recovery.md's terminal-failure rationale, its own wake-loop's logs/fix/retry sequence, and memoization.md's `mini explain` — and is now one orienting paragraph naming what is unique below it. And `reports.md`'s `### Simplification pass` was a stale copy rather than misplaced guidance: `science/references/review-passes.md` has the current version, with the `report-restructure` pass, the template check, and the read-the-diff-for-a-lost-hedge check, none of which the four-step copy carried. So it was deleted rather than moved to `writing`/`text-lint` as this item guessed, and the intro now says the science skill owns it — which also clears the orphan h3.

One repair while passing: `explain` was missing from running.md's tick-vs-read list, and `cmd_explain` only reads records, so it now sits with the safe verbs. The two new anchor links were checked with a throwaway script over every relative link and `#anchor` under `.agents/skills/`; nothing else in the tree is broken, and making that check permanent is its own item ([markdown-link-check](markdown-link-check.md)).

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/mi-ni-reference-doc-duplication.md`](https://github.com/z0u/sca2/blob/main/todo/eng/mi-ni-reference-doc-duplication.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

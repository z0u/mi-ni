---
status: done
---
# "cell" vs "condition" conflict with the glossary

Resolved the other way: "cell" collides with table/heatmap cells and Marimo cells, which can't be renamed, so prose now uses **condition** (seed-aggregated factor combination), **run** (condition × seed), and **criterion** (a hypothesis-gate clause), reserving "cell" for literal grid entities. Scheme documented in the `style-terms` skill.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/cell-vs-condition-conflict-with-glossary.md`](https://github.com/z0u/sca2/blob/main/todo/style/cell-vs-condition-conflict-with-glossary.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on; the decision is what `style-terms` here documents.

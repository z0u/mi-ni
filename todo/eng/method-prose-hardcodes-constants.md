---
status: open
tags: [reports]
opened: 2026-08-01
---
# Method prose hardcodes constants that live in the experiment module

A report's `### Schedule` cell is a literal `mo.md(r"...")`, so its numbers (anneal endpoints 50 and 90, the 2.5 opening ratio, the 0.03 hold, epochs 10/90/100) are typed rather than interpolated from the experiment module. Change a scheduler constant and the Method section describes a run that never happened. `check-templates` (`.agents/skills/report-restructure/scripts/`) cannot see this: it tracks expressions that go missing, not literals that were never expressions. Worth a lint that flags numeric literals in report prose which match a module-level constant, or simply converting the cell to an f-string.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/method-prose-hardcodes-constants.md`](https://github.com/z0u/sca2/blob/main/todo/eng/method-prose-hardcodes-constants.md) there) with the code it describes; the report is that project's ex-2.1.7.

---
status: open
tags: [cli, terminology]
opened: 2026-07-27
---
# "Cell" vs "condition" terminology split

Report prose now says "condition" for one sweep item, reserving "cell" for visual elements (heatmap/table cells). The `mini` library still says "cell" throughout (`orchestration.py`, `__main__.py` monitor output, docstrings). Decide whether to rename the library term to match — it touches CLI output and docs, so it's a deliberate rename, not a sweep-through.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/cell-vs-condition-in-mini.md`](https://github.com/z0u/sca2/blob/main/todo/eng/cell-vs-condition-in-mini.md) there) with the code it describes. The `style-terms` skill here already says "condition", so the library is the side still to move.

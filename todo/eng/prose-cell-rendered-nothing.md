---
status: done
tags: [publishing, reports]
opened: 2026-08-01
closed: 2026-08-01
---
# A prose cell rendered nothing in ex-2.1.7

The H2 verdict cell was dropped from the Markdown export, so the published report stated no verdict for H2 and carried none of the numbers that decide it. Cause: `marimo-md-export` rewrites `/// type | Title` admonitions to `!!! type "Title"` across the whole document before collecting cells, so it also rewrites `///` blocks sitting inside a cell's Python source. Cells are matched to their rendered output by MD5 of that source, so the rewrite costs the cell its output, and a hidden-code cell with no output is deleted. Only interpolated `mo.md(f"...")` cells are exposed: literal `mo.md("...")` cells are unwrapped to plain Markdown, where the transform is correct. The three other `///` blocks in that report are all in literal cells, which is why one cell went and the rest stayed. Fixed in [`scripts/export_report_md.py`](/scripts/export_report_md.py), which drives the `marimo_md_export` library directly rather than its CLI: it converts admonitions outside fenced code only, and hard-fails if any fence in the Markdown carries a source no notebook cell has. That check catches the class, not the instance — any future transform that rewrites a fence trips it. Still worth reporting upstream (MIT, ~1k lines, [jmarshrossney/marimo-md-export](https://github.com/jmarshrossney/marimo-md-export)); `inject_outputs` also builds a `warnings` list it never appends to. Note how this was caught: a structural reviewer reading the rendered document, not the source. Nothing in the source review could have found it.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/prose-cell-rendered-nothing.md`](https://github.com/z0u/sca2/blob/main/todo/eng/prose-cell-rendered-nothing.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on; the report was that project's ex-2.1.7.

---
status: open
tags: [reports]
opened: 2026-08-05
---
# Importable shared glossary for reports

Each report restates its glossary table by hand, which is how the cell/condition drift happened. A small module (e.g. `mini.reports.glossary`, or the project's own package) holding term → definition rows that a notebook imports and renders — each report selecting the terms it uses — would keep definitions identical across reports while staying self-contained when published. Needs care with memoization only if it lands in `experiment.py` inputs; keep it report-side.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/importable-shared-glossary.md`](https://github.com/z0u/sca2/blob/main/todo/eng/importable-shared-glossary.md) there) with the code it describes.

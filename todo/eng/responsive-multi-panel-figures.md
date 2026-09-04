---
status: open
tags: [reports, vis]
opened: 2026-07-16
---
# Responsive multi-panel figures in reports

Downstream, a two-panel figure (sca2's ex-2.1.1 named-pair lattice) was split into two independent `themed` figures wrapped in an inline-block row that reflows to a stack on narrow screens, with matched size via a shared projection + pinned limits + full-figure bbox rather than `sharey`. That pattern works when the panels carry no shared axis labels and share only a scale. Still undecided for wide grids (1×3 and 1×4 panels), which shrink illegibly on phones: (a) split like the lattice, but then the shared y-axis label and legend that live only on the leftmost panel have to be managed; or (b) keep them single figures and give each a declared native width (e.g. `style="--mini-fig-width: 700px"`) that a wrapper turns into a `min-width` + horizontal scroll box, so they scroll instead of shrinking below legibility (mirrors the `.report-table-scroll` rule in `docs/report.css`). Option (b) is less disruptive and generalizes; the open question is where the min-width/scroll wrapper lives — a `themed` option, or a CSS class the author opts into. Decide before more reports reuse the pattern.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/responsive-multi-panel-figures.md`](https://github.com/z0u/sca2/blob/main/todo/eng/responsive-multi-panel-figures.md) there). The `.report-figure-row` class that the lattice used lives in that project's report rather than in `report.css` here, so whichever option lands should bring its CSS with it.

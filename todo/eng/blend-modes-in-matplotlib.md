---
status: open
tags: [vis]
opened: 2026-07-21
---
# Blend modes in matplotlib figures

matplotlib has no `mix-blend-mode` — no compositing operators on artists at all. Where several series coincide (e.g. three color channels drawn over one schedule), the last one drawn wins and the rest are hidden. `mini.vis.smooth_step` sidesteps it with tapered line widths, which works but encodes an arbitrary draw order in the widths. A real multiply/screen is possible: render each series to its own RGBA buffer and composite in numpy. Two things to get right if we build it — the chrome (axes, grid, text) must be a separate layer that is not blended, or labels over- and under-expose; and each layer's empty pixels must contribute the mode's identity (1 for multiply, 0 for screen) rather than the background color, or the background gets blended in once per layer. That second one only shows up in dark mode, since empty-over-white happens to equal multiply's identity. Subline gets all of this free because SVG has the property natively ([`subline.py`](/src/subline/subline.py) sets `--blend-mode` and applies it to the series paths only) — worth revisiting if a second figure wants it.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/blend-modes-in-matplotlib.md`](https://github.com/z0u/sca2/blob/main/todo/eng/blend-modes-in-matplotlib.md) there) with the code it describes.

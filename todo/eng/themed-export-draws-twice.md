---
status: open
tags: [performance, vis]
opened: 2026-07-28
---
# `@themed` exports pay for layout and drawing twice per figure

Measured downstream while trimming one report's export time (sca2's ex-2.1.5, ~46 s after its numeric fixes), and what was left was matplotlib, structurally. `@themed` renders each figure twice (light and dark), and `base.mplstyle` sets `figure.constrained_layout.use: True`, so every save solves a layout: 9.4 s across 18 saves there, with ~13 s in `get_tightbbox` overall. On a synthetic multi-panel figure the layout engine costs about as much as drawing the figure again, and it scales with panel count rather than data volume: +0.05 s per panel (0.05 s at 1 panel, 0.56 s at 12, 1.23 s at 24), flat from 200 to 20,000 points per panel. It has to measure the rendered extent of every tick label, axis label and title to allocate margins, then iterate the solve, so a 12-panel grid pays it 12 times over.

Two things to try, in order of bluntness:

- `themed_figure_html` passes `bbox_inches="tight"` on top of constrained layout, which makes `print_figure` draw the figure a second time to measure the crop (the profile shows 36 `figure.draw` for 18 saves). Measured ~20% off a save with no visible change in what constrained layout already produced — the two are solving nearly the same problem.
- Beyond that it's the engine itself, and the options are fewer panels per figure or a fixed layout for the grid-shaped figures that don't need solving. Both change margins, so eyeball across a few reports rather than swapping blind.

A third instance of the draw-twice shape, and a fix worth knowing: a figure builder that calls `fig.canvas.draw()` to settle panel boxes before adding overlay axes, then saves, paints twice. `fig.get_layout_engine().execute(fig)` settles the same boxes (panel positions bit-identical) at a fraction of the cost: 0.47 s against 3.49 s on the figure it was found in. Any place that freezes a constrained layout mid-build can use it.

The layout claim has a bound: on a 30-panel figure carrying 1.8M scatter marks, `layout=None` saved nothing, since the cost was marker rasterization and tracked mark count (60k/panel → 3.0 s, 1k/panel → 0.25 s). So the per-panel `get_tightbbox` cost dominates only while the panels are cheap to paint; for mark-heavy figures, paint count is the thing to cut.

Reproduce with `python -m cProfile -o p.prof` around a `MINI_EXPORTING=1` `runpy` of the notebook, or monkeypatch `DefaultExecutor.execute_cell` for per-cell times. Beware absolute timings across container restarts — the box this was measured on drifted ~25% between sessions, so A/B interleaved.

## Notes

**2026-09-04, backport** — Distilled from sca2's [`todo/eng/ex-215-export-time.md`](https://github.com/z0u/sca2/blob/main/todo/eng/ex-215-export-time.md), which tracks that one report's export time; the matplotlib half is the library's, so it lives here on its own.

---
name: style-fig
description: Figure conventions for experiment reports. Data-colored marks, smooth-step sequence charts, sublines (per-token series drawn under the text), theming and dark mode, captions and nested sub-figures, plus HTML result-table and color-swatch conventions. Use when drawing or revising any figure, writing a figure or table caption, or building a results table, in a notebook.
---

A reader who has learned one figure's encoding should be able to read the next one without relearning it. The recurring panel types are packaged as helpers in `mini.vis`, whose docstrings hold the mechanics; this file says which to use when.

## Charts

A chart (loss curve, score sweep, schedule) keeps its axes. Use the stylesheet defaults from `mini.vis` and prefer meaningful ticks: a hue axis gets named ticks (Red, Green, Blue) instead of 0–1.

- Draw range bands (`fill_between`) before any summary line, or give the bands a lower `zorder`.
- Encode an _ordinal_ series (depth, size) as ordered shades of one colormap rather than categorical hues, with stops picked via `light_dark` — a colormap's dark end vanishes on a dark background.
- For per-token series, draw plateaus joined by S-curve risers with `mini.vis.smooth_step` and its band/area/marks companions (`smooth_step_marks` puts the weight on the plateaus, for a handful of discrete sites). The docstrings cover `ramp`, `breaks`, `elide`, and `fillet` (straight risers with circular corners of a given radius in points, for when the slope carries rate information).
- For all other ordinal series, use a regular line chart.
- We never use heat maps for sequences. Where the series runs over the tokens of one specific piece of text, use a subline (below) rather than either.
- Decide `sharex`/`sharey` from the units: panels measuring the same quantity share; panels measuring different quantities get their own scale, however close the numbers. Two panels with nearly-but-not-quite equal limits look like a bug.

## Geometry panels

A geometry panel shows a space (latent scatter, embedding projection) rather than a chart of one. The space is the message, so draw the domain rather than chart furniture: limits fixed from the domain — never autoscaled, since panels must be comparable across conditions and a collapsed dimension should _look_ collapsed — axes hidden, and the bound of the domain drawn instead. Equal aspect, marks and rim annotations with `clip_on=False`, and 3D projections orthographic and top-down (`ax.view_init(elev=90, azim=-90)`, `ax.set_proj_type('ortho')`, view margin 0) so the panel reads as a 2D slice.

## Color is data

Color the marks with the colors they represent; a legend or colorbar is almost always the wrong tool. Encode comparisons in the mark itself: facecolor shows the model output, edgecolor (or an inset patch, for grids) shows the true input, so an error shows as a face/edge mismatch. Loss-vs-hue lines draw as segments colored by the color at each x (round capstyle to avoid gaps).

Figures export transparent, so alpha is unreliable: several translucent copies of one color build up into a darker line, and a reader takes the extra weight for a signal. `mini.vis.color.mix(base, over, t)` pre-computes the flattened color instead, and `page_color()` gives the background the current theme assumes.

The same rule holds in prose and HTML tables: name a palette color with an inline swatch — `<span class="sw" style="--sw:#rrggbb">`, styled by `docs/report.css`, which also carries dot, outline, and themed (`--sw-light`/`--sw-dark`) variants.

## Sublines

A subline is the text itself with one sparkline per series running underneath, aligned to the tokens: `subline.subline.Subline(…).plot(tokens, series)`, whose docstring holds the mechanics. Tokens may be any width — a wide one draws as a plateau across its glyphs, the same grammar as `smooth_step`. Reach for it when the reader needs to see _which_ token a value lands on; per-character surprisal or predictive entropy over one prompt is the standing case. A matplotlib chart of the same series gives up the alignment with the glyphs, and a heatmap gives up the rate of change.

Two things are ours rather than the library's. Pass `css="svg { --bg-color: light-dark(#fff, #181c1a); }"`: its light background already matches, but its dark default is a lighter grey that reads as a box on the notebook. Then wrap the SVG with `figure_html` and externalize the group, on the same terms as any other figure.

## Result tables

Authored HTML tables (built by hand and wrapped in `mo.md`) use the shared classes in `docs/report.css` rather than inline `style=`, so central edits restyle every report at once: `report-table` on the `<table>`, `num` on numeric `<th>`s and their `<td>`s, a `report-table-scroll` wrapper for wide data, and a caption via `figure_html(..., class_="report-figure")` on the same terms as a figure. In a scored table, make it visible at a glance what counts as good: mark each column's desired direction (↑ or ↓, matching the report's glossary) in its header, and bold the values that pass their gate.

## Theming

Every figure goes through `@themed` (see `mini.vis`), which renders the plot function once per theme — its docstring explains why data gets computed outside it. Inside, pick theme-dependent values with `light_dark(light, dark)`. That includes colormaps: a light-only map's pale end disappears on dark, so pick the map itself per theme — `light_dark("RdBu_r", "berlin")` for diverging (`berlin` ships with matplotlib ≥3.11), or a `LinearSegmentedColormap.from_list` running near-background → theme accent for sequential.

Judge dark variants by compositing `_assets/<name>-dark.png` over `#111`: dark exports are transparent, and your Read tool's default matte hides both real problems and false alarms.

## Captions and sub-figures

The title goes in the caption, as its opening phrase — never in `fig.suptitle` (`ax.set_title` still names a panel _within_ a figure). A caption guides decoding ("Each column shows…") and may keep one clause of interpretation where an encoding needs it; findings and their evidence belong in prose cells near the figure. Tables get a caption on the same terms, via `figure_html`.

Panels share one matplotlib figure only when they share axes, a colorbar, or a scale the reader compares across. Otherwise render each as its own `@themed` figure with a short caption, and wrap the group in `figure_html(body, caption=..., aria_label=...)`, whose outer caption holds the shared decoding — each panel then keeps its own size and the row reflows on a narrow viewport. `report.css` styles the nesting; the docstring explains `aria_label`.

Give every figure alt text (see the alt-text skill).

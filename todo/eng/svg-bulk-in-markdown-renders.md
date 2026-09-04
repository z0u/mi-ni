---
status: open
tags: [tooling, reports, vis]
opened: 2026-09-03
---
# Inline SVGs are most of a Markdown render

`./go render` assembles a report as plain Markdown for a reader — a structural pass, a reviewer, anything that wants the document rather than the page. Figures drawn with matplotlib arrive as one `![alt](…)` line each, which is what that reader wants: the alt text says what the figure shows, and the link resolves if they want to look. Figures the report inlines as SVG (sublines, the swatch table) arrive as their full markup instead. On one downstream report (sca2's ex-2.1.1) that is 8 SVGs taking 25 KB of a 45 KB document: more than half the render is path data, sitting between the paragraphs a structural pass is there to read.

The pieces of a fix are already in place. `mini.reports.externalize_html` writes each such fragment out as a sidecar under the publisher's asset dir precisely so tooling that can't run the frontend can read it — `public/.mini/report/sublines-surprisal.html` and friends are there beside the PNGs after any render. So the render could carry a link to the sidecar where it now carries the markup. What is missing is the correspondence: one sidecar holds a group of sublines, so matching an SVG in the document to the file it came from means comparing content rather than reading a name off the tag.

Worth settling alongside it: what stands in for alt text. A `![…](…)` for an SVG group needs a description, and unlike a `themed` figure these fragments carry none today — which is its own gap, since a reader of the published page has the same problem. The `alt-text` skill is the standard; `style-fig` is where the subline conventions live.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/svg-bulk-in-markdown-renders.md`](https://github.com/z0u/sca2/blob/main/todo/eng/svg-bulk-in-markdown-renders.md) there) with the code it describes.

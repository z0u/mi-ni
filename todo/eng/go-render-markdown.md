---
status: done
tags: [tooling, reports, skills]
opened: 2026-09-02
closed: 2026-09-03
---
# Give the Markdown render a `./go` verb

`scripts/export_report_md.py` turns a notebook into a readable Markdown document, and nothing calls it: no `./go` verb, no CI step, and until 2026-09-02 no skill named it, so the only routes to it were its docstring and `tests/test_export_report_md.py`. The cost showed up in the `report-structure` pass: `report-review` said "export the document" without saying how, and the nearest export to hand was `./go preview`, which produces the HTML bundle. That bundle is about ten times the size of the same document and embeds the notebook source three times, so a pass whose point is reading the assembled document instead of the source got handed the source, buried in markup.

Naming the command in `report-review` and `report-render` fixes the immediate case. A verb — `./go render docs/m2/ex-2.2.1/report.py`, defaulting the output to `.mini/renders/<key>.md` — would make it discoverable the way `preview`, `publish` and `site` are, and give the key derivation in `docs/README.md` one implementation rather than a path each caller spells out.

Worth deciding at the same time: whether the render is stale-checked and cached like `./go preview` does for bundles (it re-runs the notebook, which is the slow part), and whether the image links should be rewritten to a resolvable path. They currently point at the notebook's live `public/.mini/`, so they only resolve from the notebook's own directory.

## Notes

**2026-09-03, tech debt** — Done, all three ways. `./go render <nb>` defaults the output to `mini.reports.render_path` (`.mini/renders/<key>.md`, the same key that names the bundle), and `report-review`, `report-render` and the `report-structure` agent now name the verb rather than the script.

Both open questions answered yes. *Stale-checked*: the mtime heuristic that `./go preview --stale-only` used was hardcoded to the bundle's `index.html` inside `scripts/export_reports.py`; it moved to `mini.reports.is_stale(nb, output)` — beside the `input_dir` it was already built on — so one heuristic now serves both renders of a report, and `--force` overrides it either way. *Links*: `localize_links` repoints each figure at the file it names, relative to wherever the render lands, and turns the surviving `<img>` tags into `![alt](…)` so every figure in the document reads the same way. It drops the publisher's `?v=` cache stamp (a browser concern; on disk it reads as part of the filename) and leaves anything that names no local file alone, warning on stderr. Verified against `docs/m2/ex-2.1.1/report.py`: six figure links, all resolving from `.mini/renders/m2/`, and the rest of the document unchanged apart from the per-render clip-path ids inside its inline SVGs.

Those SVGs are the bigger cost in a render and this didn't touch them — 8 of them are 56% of that document's bytes. Written up separately as [inline SVGs are most of a Markdown render](./svg-bulk-in-markdown-renders.md), since the fix needs a way to match a fragment to the sidecar `externalize_html` already writes for it.

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/go-render-markdown.md`](https://github.com/z0u/sca2/blob/main/todo/eng/go-render-markdown.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

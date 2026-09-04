---
status: open
tags: [tooling, reports, publishing]
opened: 2026-09-02
---
# Cut the number of on-disk copies of each figure

A report's PNGs currently exist in three places locally, and a reader (human or agent) has to know which one to reach for:

| path | written by | why it exists |
| --- | --- | --- |
| `docs/<key>/public/.mini/report/*.png` | the notebook run, via `mini.reports` | the only dir marimo's dev server serves in edit mode |
| `.mini/exports/<key>/_assets/*.png` | `./go preview` / `./go publish` | the bundle, the durable artifact, mirrored to the bucket |
| `_site/<key>/_assets/*.png` | `build_site.py --localize` | so a local preview works offline |

Each is defensible on its own and the sizes are modest (measured downstream: 5.3 MB live across three reports, 37 MB each for `.mini/exports/` and `_site/`). The cost is discernment: three plausible paths for the same file, so every skill that mentions figures has to say which, and the Markdown render's `![alt](…)` links point at the first one and therefore resolve from nowhere else.

CI is already down to one copy — `./go site --externalize` reads only the HTML and sets a `<base>` href at the bundle on the bucket. It is the local paths that multiply.

Two directions, not exclusive:

- **Make `--localize` link rather than copy.** Hardlinks into `.mini/exports/` would collapse copies 2 and 3 at no cost, since site assets are read-only once written. Note that `report-render`'s `render.py` deliberately copies rather than symlinks into its serve root — a symlinked `index.html` once let a write reach back into the marimo package — so whatever is chosen here should be a hardlink, or read-only, and the reason should be written down next to that one so the two do not read as contradicting each other.
- **Have the export source from the live dir.** `mini.reports` writes `public/.mini/` during the run and the bundler writes `_assets/` from the same figures; if the bundle could reference or move the live files instead of re-emitting them, copy 1 stops being a separate thing to know about.

Whichever way it goes, the goal is one canonical path to read a figure from, so the skills can name it once. Related: [`go-render-markdown.md`](./go-render-markdown.md), which wants the Markdown render's image links to resolve from wherever the render is written.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/one-copy-of-each-figure.md`](https://github.com/z0u/sca2/blob/main/todo/eng/one-copy-of-each-figure.md) there) with the code it describes; the sizes are that project's.

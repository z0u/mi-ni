---
name: report-render
description: View a report's figures, or export its text to Markdown. Read matplotlib or inline/JS figures and the full page in a headless browser (offline). Reports are literate scripts (mini.lit).
---

# Rendering a report to check it

## Fast path: read the figure PNGs directly (no browser)

Most report figures are matplotlib, and the publisher the runner installs (`mini.reports` + the `themed`/`light_dark` vis helpers) writes each one to disk as a real file, `_assets/<name>-light.png` / `-dark.png`, during the bundle build, regardless of the surrounding HTML. So the ergonomic way to see those figures is to build the bundle and `Read` the PNGs. No browser, no runtime, no network:

```bash
./go preview --no-serve docs/gpt-sweep/report.py   # -> .mini/exports/gpt-sweep/
ls .mini/exports/gpt-sweep/_assets/*.png           # then Read the ones you want
```

The preview also assembles `_site/`, which has a copy of each bundle's `_assets/`, so `_site/<key>/_assets/*.png` is the same file. The two differ only in `report.css` (see the gotchas).

This covers the bulk of every current report. Inline-SVG figures (e.g. `subline` sparklines) that the report wraps in `svg_figure(svg, alt_text=…, name=…)` (`mini.vis`) are likewise on disk as `_assets/<name>.html`, plain markup you can `Read` (or rasterize, below) without touching the page, and the Markdown render links each one where the markup sits in the page, with the alt text as the link's text. Reach for the browser in two cases. The first is inline/JS output without such a sidecar: it lives only inside the page's client-hydrated data island (JSON, unicode-escaped `<svg…`), so there's no file to read and the page is blank until the runtime renders it. The second is when you need the whole page (prose + figures together, layout, the show-code toggle).

A standalone `.svg` file (no browser runtime involved) rasterizes to a readable PNG without a browser via cairosvg — `libcairo`/`librsvg` are present in this env:

```bash
uv run --with cairosvg python -c "import cairosvg; cairosvg.svg2png(url='x.svg', write_to='x.png', scale=2)"
```

## Text path: read the report as Markdown

To *read* a report — prose, headings, tables and figure alt text, assembled in order — export it to Markdown. No browser, no bundle:

```bash
./go render docs/gpt-sweep/report.py      # -> .mini/lit/gpt-sweep/index.md (and index.html)
```

This weaves the document (`mini.lit`; a `# title:` header at the top of the `.py`) with no browser, writing `index.md` beside `index.html` with the figures under `_assets/`; in the Markdown they appear as `<figure>` HTML with the report's real alt text. The links resolve from the render's own directory, so `Read` follows one straight to the PNG. This is what the `report-structure` agent reads. A published literate report also serves the same woven Markdown beside its page as `<key>/index.md`.

Rendering runs the report's cells (memoized work comes from the cache), so name the reports you want; `--pdf` also prints the page. The export bundle from `./go preview` holds the same document plus provenance, thumbnails, and the PDF. Its `index.html` is several times the size of `index.md`, most of it the inline stylesheet, so reach for it only when you need the page as a page.

## Browser path: for inline/JS figures, full page, or DOM assertions

A woven bundle is self-contained: the page carries its own stylesheet (`mini.lit`'s, plus `docs/report.css` inlined by the exporter), and its figures sit beside it under `_assets/`. It renders with no network, so a network-restricted sandbox is fine.

`render.py` (beside this file) serves the bundle and drives the pre-installed Chromium:

```bash
# Get a bundle first if you don't have one: ./go preview --no-serve docs/gpt-sweep/report.py
#   -> .mini/exports/gpt-sweep/  (index.html + index.md + _assets/)
uv run python .claude/skills/report-render/render.py \
    .mini/exports/gpt-sweep -o /tmp/report.png
```

Then `Read` the PNG. `--suffix '#results'` appends to the URL; `--wait-text 'some heading'` blocks until that text renders instead of a fixed timeout.

## The PDF: what the human reads on paper or e-ink

The site build prints `report.pdf` beside each report's page (`mini.report_print`, the same print the published site links from the nav chip, reused from `.mini/pdfs/` when nothing changed), through the same engine as Chrome's print dialog, so it honours the `@page` size and `@media print` rules in `docs/report.css` (paper sized for a reMarkable 2, one section per page, tables unscrolled). The print grows the page until no section breaks across pages, then clips each page to its content, so page heights vary; `print_page(fit=False)` prints the stylesheet's fixed page instead. After editing a report, `./go preview --no-serve <report>` and hand `_site/<key>/report.pdf` to the human with `SendUserFile` so they can annotate it; the `pdf-annotations` skill reads the marked-up copy back.

`./go preview` prints the built page, which carries `report.css` from source, so a print-style edit shows in `_site/<key>/report.pdf` on the next preview; `render.py _site/<key>/ -o /tmp/report.pdf` prints it again by hand, with the page served from `_site/`. Rasterize pages with pypdfium2 (a dev dependency: `PdfDocument(path)[i].render(scale=1.5).to_pil().save(...)`) and `Read` them, or tile them into a contact sheet to see the page breaks at a glance.

To inspect one element instead of the whole page, pass a CSS selector. `render.py` shoots each match (numbering `out.png` → `out-0.png`, `out-1.png`, … when several match) after scrolling it into view:

```bash
uv run python .claude/skills/report-render/render.py \
    .mini/exports/gpt-sweep --selector 'main.lit figure' -o /tmp/fig.png
```

`main.lit` is the woven document's content column, so `main.lit figure` targets the report's figures (`main.lit table` for a table, `pre.stdout` for a cell's printed output). Tighter than a full-page shot when you only care about one figure.

## Asserting on behavior, not just looking

For visibility / layout logic, drive the DOM instead of screenshotting. `mini.report_print.served_bundle` is the reusable core (a context manager yielding the offline URL); swap the screenshot for Playwright queries:

```python
page.goto(f"http://127.0.0.1:{port}/index.html")
page.wait_for_selector("main.lit")
n_figures = page.locator("main.lit figure").count()
wide = page.evaluate("document.querySelector('main.lit table.report-table').scrollWidth")
```

## Why it works / gotchas

- Run through the project env (`uv run`; Playwright is a dev dependency), not `uvx`, so `mini.report_print` and the pinned Playwright resolve.
- The serve root is a copy of the bundle (`index.html` plus `_assets/`), never a symlink, so a write into the serve root can never reach through a link and mutate the bundle.
- Chromium: in the Claude-on-web sandbox it's pre-baked at `/opt/pw-browsers/chromium`, and `render.py` uses that if present. In VS Code or a fresh dev container it's not there (and `/opt/pw-browsers` isn't writable), so `render.py` falls back to Playwright's default resolution, and an export without any browser skips the PDF with a warning. Install it once:
  ```bash
  uv run playwright install chromium        # -> ~/.cache/ms-playwright
  uv run playwright install-deps chromium   # OS libs (libxkbcommon0, …)
  ```
  A candidate for baking into the dev container if this becomes routine; on-demand is fine otherwise (one download, then cached).
- A missing favicon/font 404 is cosmetic; the page still renders (the webfonts come from Google Fonts, so offline the fallback fonts show).
- Shared report styles (`docs/report.css`: `.sw` swatch variants, `.report-table`, the print rules) ride along two ways: inlined into each export by the exporter, and re-inlined from source at build time by `mini.reports.set_report_styles` (so central edits restyle every report without a re-export). A raw `.mini/exports/<key>/` bundle therefore carries the baked copy, so rendering it shows the styles; to preview a central edit to `report.css` without re-exporting, rebuild the site (`./go preview`) and render from `_site/<key>/` instead.

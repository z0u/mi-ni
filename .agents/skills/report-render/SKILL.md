---
name: report-render
description: Render a report in a headless browser. Works offline by bundling Marimo assets.
---

# Rendering a report to check it

A `marimo export html` bundle loads its frontend runtime (~200 JS/CSS/font URLs)
from `cdn.jsdelivr.net/npm/@marimo-team/frontend@<version>/dist`. In a
network-restricted sandbox the browser can't reach that CDN, so the page stays
blank — you can't screenshot it, and DOM assertions see nothing.

The fix: the *same* pinned `dist/` ships inside the marimo pip package under
`_static/`. Repoint the bundle's CDN refs at those local assets, serve the result,
and drive the pre-installed Chromium. No network, real render.

`render.py` (beside this file) does the whole dance — build a serve root from
marimo's `_static/` plus the bundle's CDN-rewritten HTML, serve it, screenshot:

```bash
# Get a bundle first if you don't have one: ./go export docs/gpt-sweep/report.py
#   -> .mini/exports/gpt-sweep/  (index.html + _assets/)
uv run --with playwright python .claude/skills/report-render/render.py \
    .mini/exports/gpt-sweep -o /tmp/report.png
```

Then `Read` the PNG. `--suffix '?show-code=true'` appends to the URL;
`--wait-text 'some heading'` blocks until that text renders instead of a fixed
timeout.

## Asserting on behavior, not just looking

For toggles / visibility / layout logic, drive the DOM instead of screenshotting.
`render.py`'s `_build_serve_root` + `_serve` are the reusable core; swap the
screenshot for Playwright queries. This is how the show-code default was pinned
down (PR #22) — e.g. across `?show-code` values:

```python
page.goto(f"http://127.0.0.1:{port}/index.html?show-code=false")
page.wait_for_timeout(3500)
code_shown = page.evaluate("document.body.innerText.includes('import marimo')")
page.locator("[aria-haspopup=menu]").first.click()          # open the ⋮ menu
toggle = page.locator("[data-testid=notebook-action-show-code]").count()
```

## Why it works / gotchas

- **Run through the project env** (`uv run --with playwright`), *not* `uvx`: the
  local `_static/` assets are hash-named per marimo version, so they only match a
  bundle exported by the *same* marimo. `uvx --with marimo` would resolve some
  other version and every asset would 404.
- Two asset dirs, no collision: the runtime lives under `assets/` (from
  `_static/`), the report's figures under `_assets/` (leading underscore, from the
  bundle). `render.py` symlinks both into the serve root.
- Chromium is pre-installed at `/opt/pw-browsers/chromium` (override with
  `PLAYWRIGHT_CHROMIUM`); don't run `playwright install`.
- A missing favicon/font 404 is cosmetic — the app still renders.

This same repoint-CDN-to-`_static` trick is what a full offline/archival bundle
would do at publish time; here it's just scoped to a throwaway render.

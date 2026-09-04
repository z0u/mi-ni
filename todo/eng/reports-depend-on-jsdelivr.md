---
status: open
tags: [publishing]
---
# Published reports depend on jsDelivr for the marimo runtime

`marimo export html` points ~200 `<script>`/`<link>`/font URLs at `cdn.jsdelivr.net/npm/@marimo-team/frontend@<version>/dist`, so a published report only renders while that CDN + the pinned version stay up. Not worth doing now, but for archival we could self-host `dist/` into each bundle's `_assets/` and rewrite the CDN base to a relative path in `clean_docs`/`export_reports` (same post-export surgery seam as the show-code shim). Cost: ~a few MB of JS/fonts per bundle and a maintenance tie to the marimo version. (The local half — repointing CDN refs at marimo's bundled `_static/` to browser-check an export offline — is done: see the `report-render` skill.)

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/reports-depend-on-jsdelivr.md`](https://github.com/z0u/sca2/blob/main/todo/eng/reports-depend-on-jsdelivr.md) there) with the code it describes.

---
status: open
tags: [reports, publishing, ci]
opened: 2026-09-19
---
# Print the PDFs in CI rather than at publish

`eng/publishing.md` records the decision to print each report's PDF at export, so it rides the bundle sync and is pinned with the page. Sandy's later observation (2026-09-19): the PDFs churn more than the figures do, since a print picks up every prose edit and every `report.css` change, so each publish pushes a few hundred KB of PDF into the publish tier's history that the site build could have made itself. The site build already has the resolved page and fetches the figures by URL; with Playwright's Chromium cached in the deploy workflow, it could write `<key>/report.pdf` into `_site/` and the publish tier would carry only the page and its assets.

What changes if it moves. The PDF stops being pinned: it is derived on every build from the pinned bundle plus the current `report.css`, which is what the page itself already does, so the two renditions would stop drifting apart on a stylesheet edit (today an old report's PDF restyles only on its next publish). The costs the register lists still stand and are worth measuring before deciding: a browser download per CI run (cacheable), every figure fetched on every build, and a few seconds of printing per report on every PR preview. A middle path is to print on the main deploy only and leave PR previews without a PDF. The local `./go preview` keeps printing either way, so a reMarkable review never waits on CI, and `mini.report_print` stays as the one implementation, called from the build instead of the export.

The print fetches KaTeX and the fonts through a Python-filled cache (`mini.report_print.route_remote`, 2026-09-19) because a proxied sandbox's Chromium cannot reach a CDN. A CI runner can, so the cache is a convenience there rather than a requirement.

---
status: done
tags: [reports, publishing, ci]
opened: 2026-09-19
closed: 2026-09-19
---
# Print the PDFs in CI rather than at publish

`eng/publishing.md` records the decision to print each report's PDF at export, so it rides the bundle sync and is pinned with the page. Sandy's later observation (2026-09-19): the PDFs churn more than the figures do, since a print picks up every prose edit and every `report.css` change, so each publish pushes a few hundred KB of PDF into the publish tier's history that the site build could have made itself. The site build already has the resolved page and fetches the figures by URL; with Playwright's Chromium cached in the deploy workflow, it could write `<key>/report.pdf` into `_site/` and the publish tier would carry only the page and its assets.

What changes if it moves. The PDF stops being pinned: it is derived on every build from the pinned bundle plus the current `report.css`, which is what the page itself already does, so the two renditions would stop drifting apart on a stylesheet edit (today an old report's PDF restyles only on its next publish). The costs the register lists still stand and are worth measuring before deciding: a browser download per CI run (cacheable), every figure fetched on every build, and a few seconds of printing per report on every PR preview. A middle path is to print on the main deploy only and leave PR previews without a PDF. The local `./go preview` keeps printing either way, so a reMarkable review never waits on CI, and `mini.report_print` stays as the one implementation, called from the build instead of the export.

The print fetches KaTeX and the fonts through a Python-filled cache (`mini.report_print.route_remote`, 2026-09-19) because a proxied sandbox's Chromium cannot reach a CDN. A CI runner can, so the cache is a convenience there rather than a requirement.

## Note 2026-09-19 (Fable, with Sandy) — landed

Sandy's question that settled the design: sca2 has many more reports than mi-ni and the number grows, so can the PDFs be cached in CI, keyed on the published HTML and the assets it pulls in? They can, and the assets come free: the build's printable page names every figure by a revision-pinned URL and KaTeX and the fonts by versioned ones, so a hash of the page covers them, and the only thing it misses is the tooling, which `report_print.print_stamp` adds. The memo lives in the previous `gh-pages` commit rather than the cache action: the deploy already rebuilds the whole site from state, and the site it replaces is a memo of that state, with no eviction window and no second store. `build_site.PdfMemo` does the lookup, `deploy_site.previous_site` hands each build its part of the previous deploy, and the cache action is used for the Chromium download alone. Previews keep their PDFs (each PR's set is reused across runs while its branch is unchanged) rather than the production-only middle path, since a PR that edits a report is where its PDF is worth proofing. Measured on mi-ni's seven reports: the site's prints are byte-equal to the export-era ones, and a second build prints nothing. The register entry is in `eng/publishing.md`.

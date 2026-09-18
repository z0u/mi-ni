---
status: done
tags: [reports, publishing]
opened: 2026-09-10
closed: 2026-09-18
---

# Export to PDF during review

I like to review our reports at all stages on a reMarkable 2, so I can annotate them. It would be nice if this was a standard part of our workflow: agent makes some changes; I download and annotate the PDF; I give it back to the agent to continue with.

reMarkable supports left-right swipe gesture to navigate between pages, and up-down swipes to scroll within a page. Especially during review, I would like each section to be contained in one long page, so I can swipe U/D between paragraphs, and L/R between sections.

## Note 2026-09-15

Print styles landed in `docs/report.css` (`@media print`): page as wide as the reMarkable 2 screen and about a metre tall, so each section is one long page (sections start fresh pages), 11pt body with tall lines for handwriting, tables unscrolled and wrapped, deferred figures loaded on `beforeprint`. The workflow for now is the browser's print dialog (headers and footers off) and a manual transfer. `render.py` in the `report-render` skill prints headless (`-o report.pdf`), which is the seed for a `./go` verb if the round-trip gets automated. A page has one height, so a short section leaves white space below it; that is the price of the swipe-between-sections navigation.


## Note 2026-09-16

Printing from a phone browser does not work: it ignores the `@page` size, so the reMarkable layout only comes out of desktop Chrome. The ask now is to take the browser out of the loop: the site build prints each report's PDF itself and the page links to it. Two seams. The printing is `render.py` in the `report-render` skill (headless Chromium, `-o report.pdf`), which already honours the print styles; the site build would run it per bundle and write `<key>/report.pdf` beside `index.html`, so it needs a browser in CI (Playwright's Chromium, cached like the other pinned tools). The link belongs in the nav chip `set_banner` injects (`mini.reports`), as a third entry after `← Index` and `Source`, hidden in print like the rest of the chip. Reports are pinned by revision in `docs/publish.lock`, so the PDF is per pinned bundle and never goes stale on its own.

## Note 2026-09-17

Two paging problems from the ex-2.2.9 review. A section that runs past one page breaks mid-table and mid-admonition (the H2 "Miss" callout landed alone on its own page, and a table split across the H3/H4 boundary). `break-inside: avoid` is on `.admonition` and `figure` already, so these are blocks taller than the hint can hold or a section taller than the page; the page could be taller still (the H3 section with its two figures and four tables is the size to fit), or the hint could go on the table wrapper and the callout together with the paragraph before it. Worth a check that the headless print honours the same rules as desktop Chrome, since the phone route ignores `@page`.

## Note 2026-09-17 (Fable, with Sandy)

Decided: print at publish, link at build. The build is read-only and reads only the HTML (`eng/publishing.md`), and the export session already holds the bundle, a Chromium, and the review loop, so the PDF is made there and rides the bundle sync like the thumbnails. The costs and the two things that had to hold (deterministic bytes, resolved links) are recorded with the decision in `eng/publishing.md`. Common ground with `markdown-publishing.md`: a `report.md` would take the same route, declared with `type="text/markdown"`.

## Note 2026-09-17 (Fable) — landed

The print step is in: every export (`./go preview`, `./go publish`) writes `report.pdf` beside `index.html` through `mini.report_print`, the bundle sync carries it, the bundle's head declares it (`<link rel="alternate" type="application/pdf">`), and the published page links it from the nav chip. Bytes are deterministic (dates and document ID stripped), and author links in the PDF resolve to GitHub and the site. The decision and its costs are in `eng/publishing.md`. Existing reports get a PDF on their next publish. What remains here is the paging note above (break hints on the table wrapper and the callout with its preceding paragraph; checkable on any bundle's `report.pdf` now) and the return half of the loop, which the `pdf-annotations` skill already covers.

## Note 2026-09-18 (Fable, with Sandy)

The index links each PDF too (a chip at the head of the entry's figure strip, read from the same `<link rel="alternate">`), and the export says when it is printing. The paging problem is answered from the other side: rather than a taller fixed page, the print grows the page (from twice the stylesheet's height, doubling toward the 200-inch PDF limit) until the document has one page per section, then clips each page to its ink, so a section is exactly as long as it needs to be (ex-2.2.9: 14 pages from 230 mm to 1.9 m, where the fixed page gave 18 with four sections broken). The stylesheet's 1010 mm stays for the browser's print dialog, which cannot clip. With every section on one page there is nothing left for the break hints to do in the export; they still serve the browser route. Done.

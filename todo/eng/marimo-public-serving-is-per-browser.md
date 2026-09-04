---
status: partial
tags: [publishing, reports, vis]
opened: 2026-08-19
---
# Marimo serves `public/` per browser, not per notebook

Interactive figure URLs (`public/.mini/<stem>/<name>`, from `report_bundle` in `mini.reports`) 404 for every notebook except the first one loaded in a given browser. Diagnosed 2026-08-19 on marimo 0.23.16.

The route is `GET /public/{filepath}` in `marimo/_server/api/endpoints/assets.py`, and it does not resolve the URL against the page that requested it. It reads an `X-Notebook-Id` header, loads that notebook's `AppFileManager`, and joins `filepath` onto *that* notebook's directory; with no header, or when the join misses, it raises `{"detail": "File not found"}`. The header is stamped on by a service worker (`/public-files-sw.js`), whose entire state is `let notebookIdPromise = new Promise(resolve => ...)` fed by `postMessage` from each notebook page on load. A promise resolves once, and a service worker at scope `/` is a singleton shared by every tab on the origin — so the first notebook to post wins, and every other notebook's figures are then looked up under its directory.

Confirmed by curling one server with forged headers: `X-Notebook-Id: <one report>/report.py` serves that report's PNGs and 404s another's; swap the header and the results swap. Reproducing with plain curl is misleading — no header means the same 404 by a different route through the same `raise`.

Workarounds today: one notebook per browser (unregister the service worker under DevTools → Application when switching), or a second `marimo edit` on another port, since a different origin gets its own service worker.

Two directions, neither taken yet. **Upstream:** report it; the fix is theirs and is small — key the id off the requesting client, or let the URL carry the notebook key. **Local:** move interactive renders onto marimo's virtual files (`/@file/<len>-<name>`), which carry no header and so are immune.

The local route stays inside marimo's public API. `mo.image(src=<path>)` promotes a string that passes `os.path.isfile` to a `Path`, reads the bytes, registers them as a virtual file, and returns `Html(h.img(src=<@file url>))` — a bare `<img>` when `caption is None`, a `<figure>` otherwise. So a `Publisher` subclass keeps writing the readable file to disk exactly as today, hands the path to `mo.image`, and lifts `src` back out of the tag. Refcounting cooperates: `Html.__init__` increments the count of every registered virtual filename found in its own text, so the reference is held by whichever `Html` finally carries the URL, and the one `mo.image` returned can be discarded. The requirement is that the figure output reach an `Html` (or `mo.md`) while the file is still registered — a bare `str` crossing a cell-lifecycle boundary would let `dispose()` sweep the blob at refcount zero. Check how `themed` threads its output before committing.

Costs that remain: the URL's filename is `<tid>-<8 random chars>.<ext>`, so a browser's "save image as" loses the good name (the disk copy keeps it); URLs are per-render rather than stable per name; and with no kernel context `mo.image` degrades quietly to a data URI — measured at 65,852 base64 characters for a 49 KB PNG, which is the output-size problem the publisher exists to avoid. Export must therefore keep writing real files; this replaces the interactive branch of `report_bundle` alone.

The docstrings on `public_dir` and `report_bundle` said marimo resolves `public/<path>` relative to the notebook page; corrected in the same commit as this item.

## Notes

**2026-08-19, local fix landed** — `Publisher` grew a `virtualize` flag; `report_bundle`'s interactive branch sets it, so a live render's URL is now marimo's `@file/` one. Verified end to end against a real edit-mode kernel: the URL fetches 200 with byte-identical content and *no* `X-Notebook-Id` header, while `/public/` still 404s without one. Every `@themed` decoration across `docs/` names its plot function `_`, so the URL is always consumed in the cell that minted it and the refcount requirement holds.

One thing found on the way: `virtual_files_supported` is False in an *export* kernel (`_export/file.py` passes `virtual_file_storage=None`), where `mo.running_in_notebook()` still returns True. So `_virtual_url`'s `data:` check is load-bearing rather than defensive — the notebook check alone would let a data URI through.

Still open: reporting it upstream. Nothing here fixes marimo, and the workaround costs a per-render URL and a second copy of each figure in shared memory.

**2026-08-19, the 404 warnings in the terminal** — editing a plot function can log `Failed to convert virtual file to data URI: ./@file/… Error: 404` from `dom_traversal`. Benign, as far as the evidence goes. marimo auto-exports `__marimo__/report.html` during an edit session, and that export inlines virtual files as data URIs; a re-render mints a new virtual file and disposes the old one, so a pass that fires mid-re-run can still name the disposed one. On failure the inliner leaves the `./@file/` URL in place, so the damage would be visible — and both reports' snapshots came out with zero leftover URLs and every image inlined, the last write landing 48s after the logged warning. `__marimo__/` is gitignored and regenerated, our own export path never mints virtual files (`exporting()` turns `virtualize` off), and the browser is unaffected because it fetched the live URL. Worst case is a stale snapshot showing one missing figure when the notebook is reopened, until its cell re-runs. Not chased further: the trigger is a timing window inside marimo's auto-export, and pinning it exactly buys nothing we would act on.

**2026-09-03, backport** — Ported from sca2 with the code that cites it, so some details above describe that project's tree; the reasoning is what the shared code relies on.

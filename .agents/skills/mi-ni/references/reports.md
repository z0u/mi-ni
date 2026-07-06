# Report bundles

Externalizing a report's figures and data so the exported HTML stays light and
publishes off Git LFS. This is the report author's side of storage; the `publish`
store primitive it builds on is in [storage.md](./storage.md#publishing-to-the-web),
the `themed` figure hook that feeds it is in [vis.md](./vis.md), and the *why* behind
the bundle-plus-`<base>` design is in `eng/publishing.md`.

A report is a **bundle**: one Marimo HTML document plus its heavy assets (figures,
data blobs), exported to a self-contained dir and synced to the bucket as a unit. The
report notebook (`docs/**/*.py`) is the only thing in Git; the HTML is never committed.

**Produce.** Set a `Publisher` once in the report's setup cell; every `themed` figure
then externalizes through it (figure cells are unchanged), and `asset_url` is the
general verb for any blob a report's JS reads (a large JSON for a data browser, an
SPA's data files):

```py
from mini.reports import use_publisher, report_bundle

pub = use_publisher(report_bundle(__file__))   # assets → this report's bundle dir
url = pub.asset_url(points_json, name="points.json")   # -> '_assets/points.json'
```

Each asset is written to `_assets/<name>`, **keyed by its readable name** — so the URL
is stable across re-exports and a re-render overwrites in place (nothing accumulates on
the bucket), and a browser "Save as" suggests that name (it takes the URL's last segment;
the bucket sets no `Content-Disposition`). Two *different* blobs under one name in a
report raises (give each a distinct `name=`). With no publisher, figures inline as
self-contained `data:` URIs, so a no-frills export still works.

**Publish, then build.** Two halves, split by trigger. `./go publish` (authenticated)
exports each report to `.mini/exports/<key>/` and mirrors that bundle to the bucket at
`exports/<key>/` — the heavy half (it runs the notebook, which needs the data + a write
token). `scripts/build_site.py` (read-only; CI) then *pulls* each synced bundle, resolves
author links against the repo, and inserts one `<base href="…/exports/<key>/">` in the
`<head>` so the relative `_assets/…` URLs resolve at the bucket — no per-URL rewriting,
no bucket writes. The same HTML opened locally (after `./go export`, which exports the
bundle) resolves `_assets/…` to the co-located files (offline; real PNGs), because the
build *localizes* when there's no bucket. Each report is one independently syncable
bundle, served at `<key>/`.

Because `<base>` repoints *every* relative URL, the rule is **the only relative URLs in
a report are its assets**. Author-written nav/source links would break against the
bucket, so `build_site` resolves them: a link to another report or `.md` becomes its
rendered page, a link to a source file becomes its GitHub source, and anything it can't
place is left alone with a warning. Write natural relative links
(`[experiment](./experiment.py)`); the absolute targets are derived from the git remote
(override with `MINI_SITE_URL` / `MINI_SOURCE_URL`). Design notes: `eng/publishing.md`.

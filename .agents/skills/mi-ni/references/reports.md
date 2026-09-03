# Reports

A report is a Marimo notebook (`docs/**/*.py`) that reads an experiment's durable results and renders them for the web. Two parts of making one live in the science skill rather than here: fixing what the report will claim *before* the experiment runs (the preregistration discipline), and the passes any report prose goes through once written. This file covers the rest — publishing the report as a self-contained bundle once results land.

## Report bundles

Externalizing a report's figures and data keeps the exported HTML light and publishes off Git LFS. The `publish` store primitive it builds on is in [storage.md](./storage.md#publishing-to-the-web), the `themed` figure hook that feeds it is in [vis.md](./vis.md), and the *why* behind the bundle-plus-`<base>` design is in `eng/publishing.md`.

A report is a _bundle_: one Marimo HTML document plus its heavy assets (figures, data blobs), exported to a self-contained dir and synced to the bucket as a unit. The report notebook (`docs/**/*.py`) is the only thing in Git; the HTML is never committed.

### Set `app_title`

Marimo's exported `<title>` defaults to the *filename* stem, and every report is conventionally named `report.py` (see [authoring.md](./authoring.md)), so an unset title publishes every report as "report" (browser tab, bookmark, search result: all indistinguishable). Pass a real title in the `App(...)` call, matching the report's H1:

```py
app = marimo.App(
    width="medium",
    app_title="nGPT scaling: width, depth, and learning rate",
    css_file="../report.css",
)
```

### Produce

Set a `Publisher` once in the report's setup cell; every `themed` figure then externalizes through it (figure cells are unchanged), and `asset_url` is the general verb for any blob a report's JS reads (a large JSON for a data browser, an SPA's data files):

```py
from mini.reports import use_publisher, report_bundle

pub = use_publisher(report_bundle(__file__))   # assets → this report's bundle dir
url = pub.asset_url(points_json, name="points.json")   # -> '_assets/points.json'
```

Each asset is written to `_assets/<name>`, keyed by its readable name, so the URL is stable across re-exports and a re-render overwrites in place (nothing accumulates on the bucket), and a browser "Save as" suggests that name (it takes the URL's last segment; the bucket sets no `Content-Disposition`). Two *different* blobs under one name in an export raises (give each a distinct `name=`).

`report_bundle` picks the destination from the context, because the two documents resolve relative URLs against different roots. Exporting, that's the bundle's `_assets/` beside `index.html`. Interactively, it's `public/.mini/<notebook stem>/` next to the notebook — marimo's dev server serves a notebook's `public/` and nothing else. Two things differ there: the same-name guard is off (re-running a figure cell after an edit is a replacement, not a collision), and the URL carries a `?v=<content hash>` so the browser shows the re-render instead of the copy it cached under that stable filename. Both are gitignored. Externalizing interactively too is what lets a figure-heavy report render at all: inlined, a page of `data:` URIs runs past the output size marimo will display. With no publisher at all, figures still inline, so a no-frills export works.

### Consume

A report reads durable results *by name* and must open cleanly before they exist. Resolve refs in one setup-cell helper that returns `None` when unpublished, and gate the first data cell with `mo.stop` showing the command that produces the data — every cell after it can then assume results:

```py
def load_results() -> dict | None:
    store = project_store()
    art = store.get_ref(METRICS_REF)         # ref name published by experiment.py
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        return json.loads(store.get(art, Path(d) / "metrics.json").read_text())

mo.stop(loaded is None, mo.md("No results yet — run the experiment:\n```bash\nbin/mini run …\n```"))
curves = loaded  # re-export under a new name; see below
```

`mo.stop` halts only cells *downstream of the guard cell*. Cells that read the loader's variable directly bypass the guard and crash on `None` in a data-less export — so consume the data only through names the guard cell defines (its re-export, or stats derived there), never the loader's own output.

Ref names are stringly typed: the experiment `set_ref`s them and the report `get_ref`s them, so declare them once in `experiment.py` and import them from the report (`from experiment import METRICS_REF` — Marimo puts the notebook's directory on `sys.path`). Sweep constants the report reiterates (widths, seeds) can ride along in the same import. Namespace refs by experiment (`reports/gpt-sweep/curves`) so experiments with similar names can't collide.

Provenance is automatic. While the report renders, every `get_ref` it makes is recorded by the active publisher into the bundle's `_assets/provenance.json` (ref → the producer stamped at `set_ref` time: experiment, task, git state, run time — see [storage.md](./storage.md)), and the exporter injects a folded "Data provenance" chip (bottom-left, mirroring the nav banner) citing each producing experiment. No per-report code; a report whose refs are unstamped (written before provenance existed, or outside a task worker) gets no chip until the producing step re-runs. The chip's content derives only from the refs in the store, so re-exporting unchanged data yields the same footer — publishing stays idempotent.

Quote numbers in prose as computed values (`mo.md(f"…{best:.2f}…")`), derived in the guard cell or below it, so the text can't drift from the data. And compute the stats *before* writing any qualitative claim — including figure alt text: a placeholder like "the lines nearly coincide" written ahead of the data will survive into a published report saying the opposite of what happened.

### Publish, then build

Two halves, split by trigger. `./go publish` (authenticated) exports each report to `.mini/exports/<key>/` and mirrors that bundle to the bucket at `exports/<key>/` — the heavy half (it runs the notebook, which needs the data + a write token). This is a deliberate step, *not* something experiment completion does for you: an experiment publishes its *results* to the store, but the *report* bundle ships only when you run `./go publish`, and the build silently skips a report that was never published (a warning, not an error), so the site just quietly lacks it. Publish once the report renders the results. Publishing also pins the bundle's revision (a publish-tier commit sha) in `docs/publish.lock`; commit that file with your changes, since the site serves each report at its pinned revision, so the publish deploys nothing until the pin lands on main (a PR preview serves the branch's pins meanwhile).

Publish when you open or update a PR that touches a report, once you're reasonably happy with it; it needn't be perfect. The PR preview serves the branch's pinned bundle, so publishing (and committing the bumped `publish.lock`) is what lets a reviewer see your change rendered; skip it and the preview shows the *old* report while your diff claims otherwise. Re-run `./go publish` and commit the lock bump on each round of report edits you push. `scripts/build_site.py` (read-only; CI) then *pulls* each synced bundle at its pinned revision, resolves author links against the repo, and inserts one `<base href="…/exports/<key>/">` in the `<head>` so the relative `_assets/…` URLs resolve at the bucket — no per-URL rewriting, no bucket writes. The same HTML opened locally (after `./go preview`, which exports the bundle and reassembles the site) resolves `_assets/…` to the co-located files (offline; real PNGs), because the build *localizes* when there's no bucket. Each report is one independently syncable bundle, served at `<key>/`.

Because `<base>` repoints *every* relative URL, the rule is that the only relative URLs in a report are its assets. Author-written nav/source links would break against the bucket, so `build_site` resolves them: a link to another report or `.md` becomes its rendered page, a link to a source file becomes its GitHub source, and anything it can't place is left alone with a warning. Write natural relative links (`[experiment](./experiment.py)`); the absolute targets are derived from the git remote (override with `MINI_SITE_URL` / `MINI_SOURCE_URL`). Design notes: `eng/publishing.md`.

## Verifying a rendered report from a sandboxed agent session

Headless Chromium *does* render an exported bundle here, and the report-render skill is the how. (The one catch it handles: the Marimo frontend loads from a CDN the sandbox can't reach, so a naive render stays blank; `render.py` repoints those refs at marimo's bundled `_static/` before serving.) Reach for it to screenshot a report — layout, prose, figures in situ — or to assert on client-side behavior (the show-code toggle, visibility logic) via Playwright DOM queries. Commands, DOM-driving, and gotchas live in that skill.

For a quick check you often don't need a browser at all:

- Structure: grep `index.html` — `Traceback|marimo-error` should have zero hits, and a string produced *below* the `mo.stop` guard (a computed number, a section heading) proves the data cells ran. Beware that cell *source* is embedded in the HTML too, so grep for rendered output, not code.
- Figures: `Read` the exported `_assets/<name>-{light,dark}.png` itself — faster and more faithful than a screenshot when you only care about one figure. Judge the dark variant composited over `#111` (see the style-fig skill).
- Inline SVG output (e.g. subline): if the report wraps the chunk in `externalize_html(html, name=…)` (mini.reports), the same markup is also a plain file at `_assets/<name>.html` — `Read` it instead of digging through the session JSON. To *see* it, extract the `<svg>…</svg>` and rasterize with cairosvg (`uvx --with cairosvg`), first stripping any external `@import url(...)` font rule. Glyph metrics are approximate without the webfont (text drifts relative to per-character marks), but shape and story read fine. Simpler still: regenerate the SVG standalone with the same code and data the report uses — that also exercises the figure code path. (For a *faithful* shot with webfonts, `render.py --selector '.output svg'` shoots the element straight from the rendered report — see the report-render skill.)

Need to drive Chromium for something outside a report bundle (a self-contained page)? `render.py` is the reference — copy its serve-root + Playwright setup, which already encodes the sandbox specifics (the `/opt/pw-browsers/chromium` executable, serving over localhost, `domcontentloaded` for pages that touch external hosts).

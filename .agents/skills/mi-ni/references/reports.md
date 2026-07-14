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

**Consume.** A report reads durable results *by name* and must open cleanly before
they exist. Resolve refs in one setup-cell helper that returns `None` when
unpublished, and gate the first data cell with `mo.stop` showing the command that
produces the data — every cell after it can then assume results:

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

`mo.stop` halts only cells *downstream of the guard cell*. Cells that read the
loader's variable directly bypass the guard and crash on `None` in a data-less
export — so consume the data only through names the guard cell defines (its
re-export, or stats derived there), never the loader's own output.

Ref names are stringly typed: the experiment `set_ref`s them and the report
`get_ref`s them, so declare them once in `experiment.py` and import them from the
report (`from experiment import METRICS_REF` — Marimo puts the notebook's directory
on `sys.path`). Sweep constants the report reiterates (widths, seeds) can ride along
in the same import. Namespace refs by project area (`reports/<group>/<experiment>/metrics`) so
experiments with similar numbering can't collide across groups.

**Provenance is automatic.** While the report renders, every `get_ref` it makes is
recorded by the active publisher into the bundle's `_assets/provenance.json` (ref →
the producer stamped at `set_ref` time: experiment, task, git state, run time — see
[storage.md](./storage.md)), and the exporter injects a folded "Data provenance"
chip (bottom-left, mirroring the nav banner) citing each producing experiment. No
per-report code; a report whose refs are unstamped (written before provenance
existed, or outside a task worker) simply gets no chip until the producing step
re-runs. The chip's content derives only from the store's refs, so re-exporting
unchanged data yields the same footer — publishing stays idempotent.

Quote numbers in prose as computed values (`mo.md(f"…{best:.2f}…")`), derived in
the guard cell or below it, so the text can't drift from the data. And compute the
stats *before* writing any qualitative claim — including figure alt text: a
placeholder like "the lines nearly coincide" written ahead of the data will
survive into a published report saying the opposite of what happened.

**Publish, then build.** Two halves, split by trigger. `./go publish` (authenticated)
exports each report to `.mini/exports/<key>/` and mirrors that bundle to the bucket at
`exports/<key>/` — the heavy half (it runs the notebook, which needs the data + a write
token). This is a deliberate step, *not* something experiment completion does for you:
an experiment publishes its **results** to the store, but the **report** bundle ships
only when you run `./go publish` — and the build **silently skips** a report that was
never published (a warning, not an error), so the site just quietly lacks it. Publish
once the report renders the results. `scripts/build_site.py` (read-only; CI) then *pulls* each synced bundle, resolves
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

## Verifying a rendered report from a sandboxed agent session

Don't try to screenshot the exported bundle with headless Chromium: the HTML
hydrates client-side and pulls the Marimo frontend from a CDN, which the
sandbox's browser can't reach (it doesn't inherit the agent proxy), so the page
renders blank — `document.body.scrollHeight` of 0 over both `file://` and a
localhost server. Verify without a browser instead:

- **Structure:** grep `index.html` — `Traceback|marimo-error` should have zero
  hits, and a string produced *below* the `mo.stop` guard (a computed number, a
  section heading) proves the data cells ran. Beware that cell *source* is
  embedded in the HTML too, so grep for rendered output, not code.
- **Figures:** `Read` the exported `_assets/<name>-{light,dark}.png` directly —
  faster and more faithful than a screenshot. Judge the dark variant composited
  over a dark background (e.g. `#111`), the way readers will see it.
- **Inline SVG output** (e.g. subline): extract the `<svg>…</svg>` and rasterize
  with cairosvg (`uvx --with cairosvg`), first stripping any external
  `@import url(...)` font rule. Glyph metrics are approximate without the
  webfont (text drifts relative to per-character marks), but shape and story
  read fine. Simpler still: regenerate the SVG standalone with the same code
  and data the report uses — that also exercises the figure code path.

If you must drive Chromium (e.g. for a self-contained page): the executable is
`/opt/pw-browsers/chromium` (pass `executable_path=`), pages referencing
external hosts hang `goto` (use `wait_until="domcontentloaded"`), and the
browser can't read the session scratchpad under `/tmp` — serve from the repo
tree over localhost.

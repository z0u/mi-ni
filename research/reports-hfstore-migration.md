# Moving reports off Git LFS onto HFStore

By Claude Opus 4.8

> **Update (later revision).** The shape below landed, then was refined in two ways
> that supersede parts of this note:
> 1. **No exported HTML in Git at all.** The original plan committed a light HTML shell
>    and externalized only the assets. We now commit *neither*: each report notebook is
>    the only source, exported on demand to a self-contained bundle and synced to the
>    bucket under `exports/<key>/`. The Pages build pulls the synced bundles. (Motive:
>    committed HTML makes PRs hard to review.)
> 2. **Assets are keyed by readable name, not content hash.** `_assets/<name>` (not
>    `_assets/<sha>/<name>`), so a re-export overwrites in place and the bucket doesn't
>    accumulate orphans — this resolves the "Accumulation / GC" open question for report
>    assets, and "where externalization runs" (export-time `./go publish`, read-only
>    build). The content-addressed CAS (`cas/<sha>`, refs, `publish`) is unchanged for
>    experiment *data*. Where the two differ, this update wins; the rest still describes
>    the mechanism (the relative-URL + `<base>` switch, the author-link resolver).

How we publish reports (and their figures) through the new `HFStore` instead of
Git LFS, so an agent can run an experiment and publish results end-to-end. It
started as an investigation and now records the decided architecture and the
mechanism that landed; it builds on [the artifact-store sketch](./artifact-store.md),
which called this out as step 3 ("`publish` for reports and figures; move large
assets off LFS"). The short version: externalize each report's figures/assets as
content-addressed files referenced by a **relative** URL, and flip one `<base
href>` at build time to repoint them — local files when opened off disk, the HF
bucket when served from Pages.

## Why move

The reports are self-contained Marimo HTML exports, and they live in Git LFS:
`.gitattributes` tracks `docs/**/__marimo__/**` (and `docs/**/large-assets/**`).
The `__marimo__` exports today:

| report | size | content |
| --- | --- | --- |
| `docs/__marimo__/gpt.html` | 489 KB | figure-heavy |
| `docs/__marimo__/gpt_sweep.html` | 438 KB | figure-heavy |
| `docs/__marimo__/themed.html` | 356 KB | a gallery of themed figures |
| `docs/__marimo__/subline_demo.html` | 67 KB | mostly prose/code |
| `docs/__marimo__/getting_started.html` | 65 KB | mostly prose/code |

LFS is the thing blocking end-to-end publishing. An agent in Claude Code Cloud
can't write LFS objects, so it can't commit an updated report — which means it
can run an experiment but not publish the result. The artifact-store notes list
the same four LFS problems (not writable from the Cloud, the 1 GB ceiling, not
shareable mid-flight, source not readable on GitHub); the first is the one that
stops the loop dead.

The replacement is already built and, as of this investigation, **proven
end-to-end from this environment**. A `put` → `publish` → HTTPS-fetch round trip
against the configured bucket (`z0u/mi-ni-store`) returns the bytes verbatim with
a correct `Content-Type`:

```
put ok: 1f085cef5ac7 size 38
published: https://huggingface.co/buckets/z0u/mi-ni-store/resolve/published/smoke/1f085cef5ac7.txt
fetched bytes: 38  content-type: text/plain  roundtrip match: True
```

So the egress allow-list (`*.xethub.hf.co`, `*.cdn.hf.co`) and the token are in
place here, and the only missing piece is wiring the *reports* to use it.

## The architecture: a coherent bundle + a `<base>` switch

A report is a **bundle**: one Marimo HTML document plus its heavy assets (figures,
data blobs). The assets are externalized when *produced* and referenced by a
**relative** URL — `_assets/<sha>/<name>.png` — written to a `_assets/` dir beside the
exported HTML. That one decision (relative, not absolute) is what makes the same
HTML work in every context, because *where* `_assets/…` resolves depends only on a
single `<base href>` in the `<head>`:

- **Opened locally** (off disk, or `./go serve`): no base tag, so `_assets/…`
  resolves to the co-located files. Offline, and the figures are real PNG files.
- **Published to Pages**: `build_site.py` uploads `_assets/` to the HF bucket and
  inserts one `<base href="…/resolve/published/reports/<name>/">`. Now the *same*
  relative URLs resolve at the bucket. Pages carries only the HTML; the bytes are
  served from HF's CDN.

So publishing is "upload the assets + insert one tag," with **no per-URL
rewriting** — which matters, because the asset URLs live inside Marimo's
doubly-escaped session JSON (see below) where surgical rewriting is fragile. A
`<base>` sidesteps that entirely: the relative URLs resolve at DOM-render time
against the document base, so one tag repoints all of them, *including* the ones
buried in the JSON, *and* a relative `fetch("_assets/data.json")` in an
interactive report (it resolves against `document.baseURI` too).

## Why relative + `<base>`, not a rewrite

Two things had to be true for the `<base>` switch to be safe, and both were
verified against a real export this session:

**Marimo contributes no relative URLs of its own.** A fresh `marimo export` has
**zero** in-page fragment anchors (`href="#…"`, the classic `<base>` footgun) and
**zero** incidental relative `src`/`href` — every framework resource is an absolute
`cdn.jsdelivr.net` URL, and images are data-URIs inside the session JSON. So a
`<base>` governs *only* the relative `_assets/…` references we deliberately
introduce. (Sweeps over `docs/__marimo__/themed.html`: 0 fragment links, 0
relative URLs, 194 absolute.)

**Externalizing at production beats post-processing.** Our `themed_figure_html`
(`src/mini/vis/nb.py`) renders each figure *twice* (light + dark) and previously
emitted `<img src="data:image/png;base64,…">` — the bulk of a heavy report. On
export those strings are buried in the cell-output session snapshot: JSON, inside a
`<script>`, with `json_script` escaping `<`/`>`/`&` to `\uXXXX` (and Marimo's own
matplotlib images nested one level deeper still, a data URI inside a
JSON-string-of-a-mimebundle). Trying to *extract* that base64 back out is the
fragile path. Emitting a short relative token instead is robust — and since we own
the code that emits it, there's nothing to parse. (No Marimo CLI flag externalizes
images either; confirmed in `_server/export/exporter.py`, `templates.py`,
`_output/mpl.py`.)

The **one caveat** of a document-global `<base>`: it also repoints *author-written*
relative links — a markdown `[experiment.py](./experiment.py)` — at the bucket,
where they'd 404. The convention is therefore **the only relative URLs left in a report
are store assets**. Rather than make authors hand-write absolute URLs, `build_site`
*resolves* the author links: `mini.reports.stray_links` finds the non-asset relatives
and a `LinkResolver` rewrites each to an absolute target — the rendered page for a
report/`.md` the build emits, the GitHub source for a plain source file — so it
survives the base (and stays relative in localize mode for offline nav). The absolute
roots are derived from the git remote (`MINI_SITE_URL` / `MINI_SOURCE_URL` override);
anything the resolver can't place is left alone with a warning. Conveniently, the
existing reports' relative source links were *already* dead on Pages — `build_site`
never copied `.py` files — so resolving them fixes a latent bug rather than creating
one.

### Git size is now a non-question

Earlier I estimated a de-figured report at ~50–70 KB (the figure-light
`getting_started`/`subline_demo` floor, since Marimo's scaffolding is CDN-loaded,
not inlined). With externalization that's moot: the assets aren't in Git at all
(`_assets/` is gitignored; durable home is the bucket), and the committed HTML is
just the light shell. The demo export below came out at 41 KB with its two figures
moved to files.

### A note on the bucket-hosted-HTML alternative

We considered serving the *HTML itself* from the bucket (no Pages). It's viable —
HF serves `.html` as `text/html; Content-Disposition: inline` (verified), so a
browser renders it — but relative links break there (the `resolve` URL 302s to a
signed CDN URL, so the document base becomes that signed path). Pages-hosts-HTML +
bucket-hosts-assets keeps the front door pretty and the `<base>` clean, so that's
the chosen shape. A thin Pages index linking to reports stays the project's URL.

## Beyond figures: data blobs and interactive reports

Figures are the common case, but the same split generalizes, and two harder cases
are worth designing for now rather than retrofitting:

- a report that ships a **large JSON or binary blob** that in-page JS browses
  (a sortable table, a tensor viewer, an embedding explorer); and
- an **interactive SPA visualization** that *is* the report — its own JS bundle
  plus data files.

Both reduce to the same move as a figure: `publish()` the heavy bytes, reference
them by URL, and let the browser fetch them at view time. The one new requirement
is that JS `fetch()`/`XHR` — unlike an `<img>` — is subject to CORS, so the
bucket has to allow a cross-origin read from the Pages origin. **It does.** The
bucket reflects the request `Origin` on both the `resolve` redirect and the final
CDN response (verified from this environment):

```
$ curl -sI -H 'Origin: https://z0u.github.io' <resolve URL>
access-control-allow-origin: https://z0u.github.io      # on the 302
…and the CDN 200 it redirects to:
access-control-allow-origin: *
```

So a report served from `*.github.io` can `fetch()` a published JSON straight
from the bucket CDN. That's what makes the data-browser and SPA cases work, and
it's the single fact that would have sunk them if it had gone the other way.

The design consequences:

- **One verb covers it.** A data blob is just `Publisher.asset_url(data, name=…)`
  with a `.json`/`.bin` extension instead of a figure — it writes the bytes to
  `_assets/<sha>.ext` and returns the relative URL. The report hands that URL to
  whatever JS reads it; locally it resolves to the file, published it resolves
  (via `<base>`) to the bucket. No new store surface.
- **The SPA's own bundle** can ride the same path (write the JS/CSS to `_assets/`,
  reference relatively) or come from a CDN; either way the heavy *data* is an
  asset, and the committed HTML stays a thin shell.
- **Range requests** survive the trip: the CDN 200 advertises `Accept-Ranges` and
  exposes `Content-Range`, so a JS viewer can fetch a slice of a big binary — the
  partial-access story the artifact sketch wanted from `tree` artifacts, over HTTP.
- **The local-dev caveat.** A figure resolves to a local file offline; a
  cross-origin `fetch()` only works once the assets are on the bucket (a `file://`
  fetch from a served page won't). So a *running* SPA/data report is an
  authenticated-build feature — fine, since that's the publish path anyway.

## What landed

The mechanism is built and verified end-to-end against the live bucket.

**`mini.reports.Publisher`** writes each asset to `_assets/<sha>/<name>`
and returns its relative URL; `themed` (in `mini.vis`) externalizes through it when one
is set. The SHA directory keeps it content-addressed; the readable leaf is what a
browser "Save as" suggests (the bucket sets no `Content-Disposition`, so the name has
to live in the URL). Ergonomics first — one line in the setup cell, no change to figure
cells:

```python
# setup cell
from mini.vis import themed
from mini.reports import use_publisher, report_bundle
use_publisher(report_bundle(__file__))   # assets land in this report's __marimo__/_assets

# figure cell — unchanged
@themed(alt_text='…')
def _plot(): ...
mo.Html(_plot())
```

- `report_bundle(__file__)` derives the asset dir from the notebook path
  (`__file__` resolves during `marimo export` — verified). No publisher → figures
  inline as self-contained `data:` URIs, so a no-frills export still works.
- `Publisher.asset_url(data, name=…)` is the general verb for data blobs / SPAs.

**`mini.reports`** has the publish-side string ops: `insert_base(html, href)` and
`stray_links(html)` (the lint), kept dependency-light and unit-tested.

**`scripts/build_site.py`** resolves the project store and picks a mode per report:
*localize* (no bucket → copy `_assets/` into `_site` beside the HTML) or
*externalize* (bucket → upload `_assets/` and insert the `<base>`), warning on any
stray relative link either way.

**End-to-end check** (a demo report exported with `report_bundle`): 41 KB HTML, two
relative `_assets/<sha>.png` refs (0 inline), no stray links; localize resolved
every ref to a file in `_site`; externalize uploaded to `z0u/mi-ni-store`, inserted
one `<base>`, and the *same* relative ref then resolved to a live, fetchable
`image/png`. `docs/probe/report.py` is converted as the worked example (it used to
inline *and* separately publish; now it externalizes and its source links are
absolute, per the convention).

## Status: landed

The migration is complete; the items this section used to list as pending are done
(and the "commit the light HTML" framing was superseded by the *no-HTML-in-Git*
revision in the update banner above):

- **Reports are notebooks only.** No exported HTML in Git; `.gitattributes` carries
  no LFS rules. Each `docs/**/*.py` report exports on demand to a bundle synced under
  `exports/<key>/` (see `docs/README.md`).
- **`publish-docs.yml`** runs the read-only build (`uv sync --group pages` for the
  full project, `HF_TOKEN` + `MINI_STORE_BUCKET` from CI) — the export+upload heavy
  half is the agent's `./go publish`. So the CI-vs-export-time question is decided:
  authenticated export-time publish, read-only build in CI.
- **A Pages index** (`docs/index.md`) links to every report.

Deferred (tracked elsewhere, not in this note):

- Purge old LFS blobs from history (`git filter-repo`) — cosmetic; the working tree
  is already LFS-free.
- The private-CAS / public-publish two-bucket split — `todo.md`.
- **Asset GC.** Re-export writes name-keyed assets that overwrite in place, so a
  given report doesn't accumulate; the broader CAS-only-grows sweep is `todo.md` /
  issue #15.

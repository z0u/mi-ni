# Moving reports off Git LFS onto HFStore

By Claude Opus 4.8

An initial investigation: what's involved in publishing reports (and their
figures) through the new `HFStore` instead of Git LFS, so an agent can run an
experiment and publish results end-to-end. It builds on
[the artifact-store sketch](./artifact-store.md), which already called this out as
step 3 ("`publish` for reports and figures; move large assets off LFS").

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

## The shape of the migration

The plan from the sketch holds: **split at size**. Heavy bytes (figures) go to
the bucket via `publish()` and are referenced by `https://` URL; the lightened
report HTML — now mostly prose, code, and a little scaffolding — goes into Git,
where it's diffable and readable on GitHub. Keep GitHub Pages as the front door;
the served HTML just pulls its images from the bucket's CDN.

Concretely, the moving parts:

1. **Externalize figures** (the substance — see the next section).
2. **`.gitattributes`**: drop both LFS rules once figures are externalized.
   Large static assets, if we ever have them, go through `publish()` too, so the
   `large-assets/**` rule retires alongside `__marimo__/**`.
3. **`publish-docs.yml`**: drop `lfs: true` from the checkout. The build
   (`build_site.py`) is unchanged — it copies the HTML, which now carries bucket
   URLs; the viewer's browser fetches the images directly from the public bucket,
   so Pages needs no token.
4. **Re-export the five reports** with figures externalized and commit the
   lightened HTML. (`themed`/`subline_demo`/`getting_started` are standalone;
   `gpt`/`gpt_sweep` need their experiment results present to re-run.)
5. **History**: the old LFS blobs stay in history harmlessly but count against
   the LFS quota. Purging them is a `git filter-repo` rewrite — heavy, separable,
   and I'd defer it rather than block the migration on it.

## Will the lightened reports be small enough for Git?

Your intuition is right, with a caveat worth stating precisely. The two
figure-light reports (`getting_started`, `subline_demo`) sit at ~65 KB *with*
Marimo's scaffolding, and that scaffolding is near-identical across every export
(the frontend JS/CSS is referenced from a CDN, not inlined — which is why the
floor is tens of KB, not megabytes). So a de-figured `gpt.html` should land in
roughly that ~50–70 KB range too, and because the boilerplate is shared,
Git's zlib + delta compression packs the set down hard. A handful of ~60 KB
mostly-boilerplate HTML files is a non-issue for Git.

The caveat: this is an estimate from the two small reports as a proxy floor, not
a measurement of a de-figured heavy report (I can't re-export the experiment
reports here without their run results). Worth confirming on the first real
prototype — but I'd be surprised if it came out badly.

## Externalize at production, not as a post-process

This is your open question, and the Marimo internals settle it. **Externalize
when the figure is produced, not by post-processing the exported HTML.**

The reason is where the bytes end up. Our own `themed`/`themed_figure_html`
(`src/mini/vis/nb.py`) renders each figure — *twice*, a light and a dark variant
— to a base64 PNG and emits `<img src="data:image/png;base64,…">`. That's the
bulk of a heavy report (two inline PNGs per figure). When Marimo exports, that
HTML string is the cell's *output*, and it gets buried inside the session
snapshot: a JSON object, inside a `<script>`/`<marimo-config>` block, with
Marimo's `json_script` escaping `<`/`>`/`&` to `\uXXXX`. A figure from Marimo's
own matplotlib formatter is nested even deeper — a data URI inside a
JSON-string-of-a-mimebundle inside the session JSON.

So a post-processing pass would have to **parse that doubly-nested, escaped JSON
out of a `<script>` tag and reconstruct it** — a regex over `src="data:…"` won't
find it. That's fragile against Marimo's version-specific internal layout, and
there is no Marimo CLI flag that externalizes images for us (only `--include-code`
and friends; `html-wasm` externalizes the *framework* assets, not notebook
images). Confirmed by reading the installed `marimo` package
(`_server/export/exporter.py`, `_server/templates/templates.py`,
`_output/mpl.py`).

Producing externally sidesteps all of that, because **we already own the code
that emits the data URI**. The change is local to the figure-rendering boundary:
instead of emitting `<img src="data:…">`, `put()` the PNG and emit
`<img src="https://…bucket…/resolve/published/…">`. The report HTML then carries
URLs (~100 bytes each) instead of two ~60 KB blobs per figure, and the figure
bytes flow through the same content-addressed `publish()` path everything else
uses — idempotent, deduped, version-independent. `docs/probe/report.py` already
does the `put`+`publish` half (lines 121–122); it just *also* inlines the figure
via `mo.Html`, so today it pays for both. The migration is to emit the URL
*instead of* the inline image, not in addition.

Mechanically I'd add an opt-in publish path to the vis helper rather than
overload `themed`: a small wrapper that takes the figure(s), a store, and a
logical path, publishes each variant, and returns the themed `<img>` markup
pointing at the two URLs. The report cell already has the store in hand
(`LocalApparatus(NAME).store()`), and during `./go run` export the ambient store
resolves to `HFStore` whenever the bucket + token are set — so this works inside
the existing export command with no new plumbing.

### The local-fallback wrinkle

One thing to get right: `default_store` falls back to `LocalStore` when there's
no bucket/token (a fresh checkout, someone trying the repo, a `./go build` with
no auth). `LocalStore.publish` returns a `file://` URL, which won't render once
the HTML is served from Pages. So the production-time helper needs a policy: when
the resolved store can't produce a web URL, **fall back to inlining** the data
URI as today. That keeps offline/no-auth builds working and renderable, and only
the authenticated path (CI, the Cloud agent) produces lightened, externalized
reports — which is exactly the path that needs to.

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

- **One verb covers it.** A data blob is just `publish()` with a `.json`/`.bin`
  extension instead of `.png`. The vis helper exposes this as
  `Publisher.asset_url(data, name=…)` (see below) — the report publishes the blob,
  gets a URL, and hands that URL to whatever JS reads it. No new store surface.
- **The SPA's own bundle** can ride the same path (publish the JS/CSS, reference
  by URL) or come from a CDN; either way the heavy *data* goes through `publish`,
  and the report HTML committed to Git stays a thin shell. This is the `html-wasm`
  shape Marimo already uses for its framework assets, pointed at the bucket.
- **Range requests** survive the trip: the CDN 200 advertises `Accept-Ranges` and
  exposes `Content-Range`, so a JS viewer can fetch a slice of a big binary rather
  than the whole thing — the partial-access story the artifact sketch wanted from
  `tree` artifacts, now over HTTP.
- **The local-dev caveat sharpens here.** A figure degrades gracefully (inline)
  when there's no web store; a *data fetch* can't inline into an `<img>`, and a
  `file://` URL won't satisfy a cross-origin `fetch()` from a served page. So the
  interactive cases genuinely need the bucket to be live — `asset_url` returns the
  `file://` URL for local opening, but the honest position is that SPA/data
  reports are an authenticated-build feature, not an offline one. Worth a clear
  error/warning when a report calls `asset_url` and the store can't serve.

## The vis helper (landed)

`mini.vis` now has a `Publisher` and `use_publisher`, and `themed` grew an opt-in
`publish` path. Ergonomics were the priority, so the common case is one line of
setup and **no change to existing figure cells**:

```python
# setup cell — point the report's figures at the artifact store
from mini.vis import themed, use_publisher
use_publisher(LocalApparatus(NAME).store(), prefix=f'reports/{NAME}')

# figure cell — unchanged; it now externalizes instead of inlining
@themed(alt_text='…', name='loss-curve')   # name → stable URL; omit → content hash
def _plot(): ...
mo.Html(_plot())
```

- **Externalize-or-inline.** With a web-serving store each PNG (light *and* dark)
  is `put` + `publish`ed and the `<img>` carries the URL; with no store, or a
  local `file://`-only store, it inlines as before — so offline builds still
  render. The decision is per-`<img>`, in `Publisher.png_url`.
- **`asset_url(data, name=…)`** is the general-purpose verb for the data-blob and
  SPA cases above — publish arbitrary bytes/a file under the report's prefix, get
  a URL back.
- **Verified end-to-end** against the real bucket: a themed figure that inlined at
  ~27 KB of base64 now emits ~1.2 KB of HTML with two PNG URLs that serve as
  `image/png`. `docs/probe/report.py` is converted as the worked example (it used
  to inline *and* separately publish — now it just externalizes).

Tests cover the inline default, the web-URL path, the `file://` fallback, named
vs. content-hash slugs, the `use_publisher` default, and `asset_url`.

## What's left

1. ~~The vis helper~~ — done (above).
2. Convert a figure-heavy report (`gpt` or `gpt_sweep`) and **measure** the
   de-figured HTML size to confirm the Git-size estimate.
3. Drop the LFS rules + `lfs: true`, re-export the five reports, commit the
   lightened HTML.
4. Defer (separable): purging old LFS blobs from history; the private-CAS /
   public-publish two-bucket split already tracked in `todo.md` (the bucket is
   public today, so `publish` is fine, but the durable CAS is public too).

## Open questions

- **Accumulation / GC.** Every re-export publishes new content-addressed figure
  blobs; old ones linger. This is the same CAS-only-grows problem `todo.md`
  already flags for the store at large — reports just add a steady trickle. Not a
  blocker, but reports make the refcount-and-sweep need concrete sooner.
- **Stable vs. content-addressed published paths.** Publishing under a
  content-hash path means a report's image URLs churn on every re-render (fine,
  the HTML is regenerated too). Publishing under a stable `reports/<exp>/<fig>.png`
  path keeps URLs constant but is last-writer-wins. The probe report uses the
  stable form; I'd keep that for reports (the URL is meant to be shareable) and
  rely on the CAS underneath for immutability of *content*.
- **Measuring the real de-figured size**, per above — the one number this
  investigation estimates rather than measures.

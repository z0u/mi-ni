# Publishing reports to the web

*Part of the [engineering notes](./README.md).*

## Publish is a separate, outward-facing verb

`put` persists; `publish` deliberately exposes. **Public exposure is never a side
effect of persisting a result** — that separation is the point.

A CAS blob at `cas/<sha>` has no extension, so a browser gets `application/octet-stream`.
`publish` does a server-side copy *by xet hash* (instant, no bytes moved) to an
extensioned path; the bucket's resolve URL then sets `Content-Type` from that extension
and serves `Content-Disposition: inline`. So one bucket is both the durable store and a
CDN-backed asset host.

## Two tiers, split by write concurrency (#38)

By default **one bucket** backs both the CAS and the published views. That's fine until
you persist something that shouldn't be world-readable: HF buckets have **no per-prefix
ACL** (public/private is bucket-level only), so a public bucket makes *every* `put`
world-readable, not just what you `publish`. The fix is two stores — and the seam
between them is **write concurrency**, not just visibility:

| Tier | Namespaces | Writer | Concurrency | Backend |
|---|---|---|---|---|
| **Durable store** | `cas/<sha>`, `refs/` | workers | high, parallel | a **bucket** (no git history → concurrent writers never 412) — make it **private** |
| **Publish tier** | `published/`, `exports/` | a driver / CI | low, single-writer | a **dataset repo** (real git history → versioned, citable) — **public** |

The CAS *must* stay a bucket: that's the whole reason buckets were chosen (concurrent
result writers would 412 on a git-backed repo's shared parent). But `publish` and report
export are deliberate, single-writer, driver-side acts — so they can afford a git-backed
**dataset repo**, which buys what a bucket can't:

- a **public** face over a **private** CAS (the two-store split HF's bucket-level ACL forces), and
- **versioned names with history** — a citation pins to `…/resolve/<commit-sha>/published/<path>`, which a mutable bucket path can't guarantee.

So one move settles both halves of #38: point the publish tier at a dataset repo and it's
public *and* citable.

**Cost.** Not a byte more of storage. Xet chunk dedup is **account-wide**, so a blob
published from the private CAS into the public repo references the *same* chunks —
publishing pulls the blob local (the warm cache usually has it) and uploads, but Xet skips
every chunk the CAS already stored, so it's a metadata + git-commit op, not a re-transfer.
The only real deltas: public storage is best-effort quota (vs private's ~100 GB
guaranteed), and each publish is a commit rather than the in-bucket instant by-hash copy.

**Enabling it.** Set `[tool.mini] publish-repo = "<ns>/<repo>"` (or `MINI_PUBLISH_REPO`)
and flip the existing bucket to private. Unset, everything stays in the one bucket — the
seam is opt-in and `HFStore` routes `publish`/`export_*` on `publish_repo` alone, so no
experiment or report code changes. Provisioning is typically just **one** new public
dataset repo; a second public *bucket* is only needed if you want high-churn public assets
that shouldn't be versioned (reports overwrite by name, so they don't accumulate — you
usually don't).

## Reports are a bundle plus a `<base>` switch

A report is one Marimo HTML document plus its heavy assets. Assets are externalized
*at production* and referenced by a **relative** URL (`_assets/<name>`). A single
`<base href>` in the `<head>` decides where those resolve:

- **Opened locally / offline** → no base tag → `_assets/…` resolves to co-located files
  (real PNGs).
- **Published** → `build_site` inserts one `<base href="…/exports/<key>/">` → the *same*
  relative URLs resolve at the bucket CDN.

The `<base>` is what makes this work without **per-URL rewriting**, which matters because
the asset URLs are buried in Marimo's doubly-escaped session JSON where surgical rewrites
are fragile — one tag repoints all of them, including a relative `fetch()`. It's safe
because a fresh `marimo export` contributes *zero* relative URLs of its own (every
framework resource is an absolute CDN URL), so a `<base>` governs only the assets we
introduce. The one caveat: `<base>` also repoints *author-written* relative links, so
`build_site` resolves those to absolute targets (`stray_links` / `rewrite_links`) — a
link to another report becomes its rendered page, a link to a source file its GitHub
source. **Convention: the only relative URLs in a report are its assets.**

Two further decisions:

- **No HTML in Git.** The notebooks (`docs/**/*.py`) are the only source of truth; each
  exports on demand to a bundle synced under `exports/<key>/`. This keeps PRs reviewable
  *and* lets a Cloud agent publish — the original blocker was Git LFS, which a Cloud
  session can't write, so an agent could run an experiment but not publish it.
- **Assets keyed by readable name**, not content hash, so a re-export overwrites in place
  and a report accumulates no orphans (the name is also what a browser "Save as"
  suggests, since the bucket sets no `Content-Disposition`).
- **Publish/build split by trigger.** `./go publish` (authenticated, runs the notebook,
  writes the store) is the heavy half; the CI build is **read-only** — it pulls each
  bundle, resolves links, inserts the `<base>`, and never writes or runs a notebook, so
  a read-only token suffices. When the publish tier is split off (#38) the build reads
  exports straight from the public dataset repo and never touches the CAS, so CI needs
  only `MINI_PUBLISH_REPO` — not the (now private) bucket. `store_for` builds a CAS-less
  store from a `publish-repo` alone for exactly this; the single-bucket default still
  serves the build off `MINI_STORE_BUCKET`. Set whichever your store layout uses.
- **PR previews ride the same read-only build.** Because `./go publish` runs on the
  agent's branch *before* the PR opens, the bundles are already on the publish tier when
  review starts — a preview needs no compute, just HTML assembled around them.
  `pr-preview.yml` runs `./go build` on the PR and deploys to `pr-preview/pr-<n>/` on the
  `gh-pages` branch (torn down on close, linked from a sticky PR comment). That forces
  the production deploy to be branch-based too (`clean-exclude: pr-preview/`): an
  artifact deploy replaces the whole site, so previews couldn't coexist with it. Viewing
  a bundle straight off the dataset repo is *not* an option — HF serves repo HTML as
  `text/plain` behind a sandbox CSP. Known gap: exports aren't branch-namespaced, so
  re-publishing an existing key from a branch swaps the assets under the live site's
  older HTML; if that bites, publish PR exports to a `pr-<n>` git *revision* of the
  dataset repo and point the preview `<base>` at `resolve/pr-<n>/` (see todo.md).
- **Provenance is injected at export, not build.** The provenance footer (which
  experiment runs produced the data a report shows) needs the producers stamped on the
  store's refs — state only the authenticated export half can see, and only *while the
  notebook runs* (the resolutions happen inside cells). So the render records each
  `get_ref` into the bundle's `_assets/provenance.json` via the active publisher, and
  the exporter injects the footer into the HTML before the bundle syncs; the read-only
  build stays dumb. Everything in the footer derives from the store's refs — no build
  timestamps — so re-publishing unchanged data produces the same bundle content and
  publishing stays idempotent.

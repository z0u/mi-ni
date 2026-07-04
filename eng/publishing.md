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

Today **one public bucket** backs both the CAS and the published views, because HF
buckets have **no per-prefix ACL** — public/private is bucket-level only. A private CAS
+ public publish bucket is a genuine two-bucket split, and a citation-grade *versioned*
publish tier would want a dataset repo (real git history) rather than a bucket. Both are
deferred — **#38**; `HFStore` is kept backend-swappable so the publish tier can move
later without touching experiment or report code.

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
  writes the bucket) is the heavy half; the CI build is **read-only** — it pulls each
  bundle, resolves links, inserts the `<base>`, and never writes the bucket or runs a
  notebook, so a read-only token suffices.

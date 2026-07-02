# Design: storage, artifacts, and publishing

The durable rationale behind `mini.store` and report publishing — the *why* that
isn't obvious from the code, written for someone (maybe you) returning to this repo
cold. The feasibility studies and migration logs that used to live in this directory
are gone; this is the distilled conclusion. The skill (`.agents/skills/mi-ni/`) is the
*how*; this is the *why*.

## Artifacts are handles, not paths

A memo result is the small thing — a dict of metrics, a handle. The large bytes a
step produces (an activation cache, an eval dump, a figure) go in the **artifact
store**, not in the result and not as a bare volume `Path`.

Returning a `Path` pickles a *location* into the result, and that location lives in a
volume that may have evaporated by the time another process, another experiment, or a
report reads it back. Instead a step `put`s its bytes and gets back an `Artifact` — a
small, location-free handle (sha256, size, name). Blobs are content-addressed at
`cas/<sha256>` and immutable.

Three things fall out of content-addressing outputs, and they're the reason for the
whole design:

- **Durable results.** A handle carries no location, so the result pickles durably and
  resolves from anywhere that can reach the store.
- **Stable downstream memo keys.** The memo key is the task's identity,
  `fn + fingerprint(args)`. Passing a `Path` into the next step fingerprints it
  *by location*; passing an `Artifact` fingerprints it *by content*, so a
  consumer's key only moves when the bytes actually change.
- **Dedup, and idempotent `put`.** Identical bytes coincide; `put` hashes first and
  skips the upload if the blob is already present, so re-runs and cross-step duplicates
  are free.

## A CAS plus a small mutable ref layer (git's objects and refs)

The `cas/<sha>` blobs are immutable. Over them sits a small **mutable ref layer** that
names views — exactly git's objects-and-refs split. One ref layer covers three needs:
cross-experiment handoff by a stable name, `publish`'s named views, and
checkpoint pointers.

**Files vs. trees.** A directory becomes a *tree* artifact: each file is its own blob
and the handle carries the manifest. That gives per-file dedup and lets a consumer
resolve one shard without pulling the set. Reach for a tree when random access or
partial dedup matters; a single file is fine otherwise. (But see the chunked-data
non-goal below — a tree is for a handful of shards, not thousands of tiny chunks.)

## Scoping: store project-wide, memo and volume per-experiment

The artifact **store is one per project** — the sharing surface. The **memo store and
volume stay per experiment** — the isolation boundary. So cross-experiment reuse today
is **explicit**: experiment A `set_ref`s an artifact under a stable name, experiment B
`get_ref`s it — no recompute, no shared volume (`docs/acts` → `docs/probe`).

This was a deliberate fork from the original plan (one shared volume + *implicit*
cross-experiment memo dedup, i.e. B silently memo-hits A's identical prep). We chose
**keep volumes isolated, make the artifact store the sharing surface, hand off by
explicit ref**, because the implicit version drags the whole control plane into
project scope: with a shared store, `cancel`/`retry`/budget teardown and the `__run__`
metadata must all become tag-scoped, or a single `mini status` on an over-budget
experiment cancels *every* experiment's in-flight tasks. Per-experiment stores make
that scoping fall out for free, and explicit refs capture most of the value at a
fraction of the blast radius. Implicit dedup remains a deferred option — **#37**.

## Why Hugging Face buckets

A bucket is a Xet-backed repo type, **mutable, with no git history**: you overwrite in
place, and there are no commits to conflict. That last property is load-bearing —
independent workers can write results concurrently without a coordination step, where a
git-backed dataset repo would 412 on the shared parent. Immutability of `cas/<sha>` is
therefore *our* discipline (write-once-by-hash), not the backend's guarantee.

**The latency floor shapes the whole API.** Every bucket read or write pays a fixed
~2–3s round trip before any bytes move (the Xet handshake + commit), independent of
size; throughput above the floor is ~10–35 MB/s. Consequences:

- **Batch or parallelize writes.** Eight serial commits took ~15s; one batched commit,
  or eight *concurrent* commits, ~2–2.6s. The floor is overlappable latency, not a
  server-side lock — so workers write independently and we batch when files are already
  in hand.
- **Resolve trees with a thread-pool fan-out**, or resolving a tree's shards one at a
  time serializes the floor.
- **Keep checkpoints on the volume, not the CAS** (see below) — content-addressing
  mutable, superseded state pays the floor repeatedly for nothing.

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

## Non-goals and recorded decisions

- **Don't grow the CAS for chunked datatrees** (Zarr, activation dumps — trees of
  *thousands* of tiny chunks). The per-file-blob + manifest design is right for a handful
  of `.npy` shards but wrong here: N round trips on write, and a `get` reassembles the
  *whole* tree before a consumer touches one slice. Don't mirror by mutable name (that
  reintroduces the half-written-read and dual-namespace-GC hazards the CAS exists to
  avoid) and don't bake Zarr sharding into `put` (too tied to one format). For genuine
  random-access-over-network, **skip the CAS for that artifact** and let the array
  library do remote IO straight against the bucket (Zarr v3 + `s3fs`/`hf://` fsspec over
  the bucket's HTTP endpoint; verify the endpoint honours HTTP `Range` first). The CAS
  stays bytes/files/trees-agnostic, for immutable artifact handoff.
- **Checkpoints live on the volume, not the CAS.** Mid-step state is mutable, superseded
  on the next write, and resume finds "the latest for this step" by a stable *name*, not
  a hash you only learn after writing. mini auto-commits the volume only at the step
  boundary, so call `volume.commit()` right after writing an expensive checkpoint. The
  exception is a cross-process handoff where the volume isn't shared (laptop → Cloud,
  local → Modal) — there a checkpoint wants a ref in the durable store.
- **`obstore` doesn't help.** It supports only S3/GCS/Azure; bridging via fsspec forfeits
  the native-Rust speed that's its whole point, and `hf_xet` already does parallel
  chunked transfer under `HfApi`. Revisit only if mini targets those clouds directly.

## Operational constraints worth remembering

- **Egress allow-list.** Bucket I/O needs `*.xethub.hf.co`; serving figures needs
  `*.cdn.hf.co`. Without them, metadata calls to `huggingface.co` succeed while every
  byte transfer hangs on a 403 — a confusing failure mode.
- **Modal gRPC TLS.** Modal's client builds its trust store from `certifi` alone; behind
  a TLS-inspecting proxy it needs the system CA folded in
  (`mini._tls.ensure_grpc_trusts_system_ca` already does this).
- **CORS / Range.** The bucket reflects the request `Origin` on both the resolve redirect
  and the CDN response, and the CDN advertises `Accept-Ranges` — so a Pages-served report
  can `fetch()` a published JSON cross-origin and Range-slice a big binary.

## Open / deferred

- Implicit cross-experiment memo dedup, + optional shared working volume and a
  `materialize` front door — **#37**.
- Private-CAS / public-publish two-bucket split; citable versioned publish tier — **#38**.
- Wire the artifact store through the interactive `app.map`/`arun` path — **#39**.
- GC across the CAS, control plane, and Volume run dirs — **#15**.
- Smaller, unscheduled: a streaming "promote-to-hash" `put` for chunk-by-chunk writers
  (hash incrementally on a working path, then server-side copy-by-xet-hash each chunk into
  `cas/<sha>`); whether `Store` should be async to match `Volume`; the on-disk tree
  manifest format if it ever needs to be read without Python; a `mini publish` /
  artifact-aware CLI surface if reports want one.

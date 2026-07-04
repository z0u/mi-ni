# Non-goals and recorded decisions

*Part of the [engineering notes](./README.md).*

## Non-goals

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
- **The HF cache is a third storage tier, kept separate** (#50): Store = durable
  project-scoped artifacts; per-experiment Volume = working dir + checkpoints (the
  isolation boundary); HF cache = a *disposable* read accelerator for upstream weights.
  On Modal it's one workspace-wide `mini-hf-cache` Volume with `HF_HOME` pointed at the
  mount — which covers both HF sub-caches (`hub/` snapshots and `xet/` chunks) with zero
  call-site changes. It's deliberately *not* routed through the `Volume` ABC or the
  per-experiment Volume (scope mismatch), and needs no commit discipline: concurrent
  writers at worst duplicate a download. Locally the tier doesn't exist —
  `~/.cache/huggingface` already persists. Note the HF **hub** cache doesn't apply to
  buckets at all (buckets aren't a repo type; `download_bucket_files` streams straight to
  the destination) — `HFStore` has its own warm cache for that.
- **The worker's `HFStore` warm cache lives on container-local disk, not the mounted
  Volume** (`WORKER_STORE_CACHE`). Under the mount it was committed alongside results,
  so every bucket artifact grew a second, redundant copy on the per-experiment Volume.
  Ephemeral is the point: the bucket is the durable copy; the tradeoff is a bucket
  round trip if a *later container* re-reads bytes this experiment wrote (rare — the
  client-side cache still keeps whole files on local disk, where re-reads actually
  happen). A cross-container sha-index (e.g. a `modal.Dict`, self-expiring ~1 week) was
  considered and deferred: with `put` batching and Xet chunk dedup, the residual cost is
  one existence probe per new blob per container.

## Open / deferred

- Implicit cross-experiment memo dedup, + optional shared working volume and a
  `materialize` front door — **#37**.
- Private-CAS / public-publish split + citable versioned publish tier — **#38**. Landed as
  an opt-in seam: `HFStore` routes `publish`/`export_*` to a Hugging Face **dataset repo**
  (versioned → citable) when `[tool.mini] publish-repo` is set, leaving `cas/`+`refs/` in a
  (now privatable) bucket; unset keeps the single-store default. Rationale (why the split
  is by *write concurrency*, and why it costs no extra storage under account-wide Xet
  dedup) in [Publishing](./publishing.md). Remaining: provisioning (a public dataset repo +
  flipping the bucket private) and a live round-trip against a real repo.
- Wire the artifact store through the interactive `app.map`/`arun` path — **#39**.
- GC across the CAS, control plane, and Volume run dirs — **#15**, shipped: `mini gc
  <name>` (both backends) and `mini gc --store` (CAS mark-and-sweep). See
  [Reclaiming storage](./gc.md). Only the two-bucket split (#38) would reshape what
  "CAS" means here.
- Smaller, unscheduled: a streaming "promote-to-hash" `put` for chunk-by-chunk writers
  (hash incrementally on a working path, then server-side copy-by-xet-hash each chunk into
  `cas/<sha>`); whether `Store` should be async to match `Volume`; the on-disk tree
  manifest format if it ever needs to be read without Python; a `mini publish` /
  artifact-aware CLI surface if reports want one.

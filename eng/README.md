# Engineering notes

The durable rationale behind `mini`'s storage, artifact, and publishing internals — the
*why* that isn't obvious from the code, written for someone (maybe you) returning to this
repo cold. The feasibility studies and migration logs that used to live here are gone;
this is the distilled conclusion. The skill (`.agents/skills/mi-ni/`) is the *how*; these
notes are the *why*.

Start with whichever question you're holding:

- [Artifacts and the content-addressed store](./artifacts.md) — why a step returns an
  `Artifact` handle instead of a volume `Path`, the CAS-plus-refs layering, and how
  storage is scoped (the store is project-wide; the memo store and volume are
  per-experiment).
- [The storage backend: Hugging Face buckets](./storage-backend.md) — why a bucket, and
  the ~2–3s latency floor that shapes the whole batch-or-parallelize API.
- [Publishing reports to the web](./publishing.md) — why `publish` is a separate,
  outward-facing verb from `put`, and how a report bundle reaches the web through a
  single `<base>` switch.
- [Reclaiming storage: `mini gc`](./gc.md) — the three durable planes every experiment
  leaves behind, and the mark-and-sweep that reclaims the CAS without ever collecting a
  live memo hit.
- [Non-goals and recorded decisions](./decisions.md) — chunked datatrees, checkpoints,
  `obstore`, the HF cache tier, and the open/deferred list with issue links.
- [Operational constraints](./operations.md) — the egress allow-list, Modal gRPC TLS,
  and CORS/Range: environment facts that cause confusing failures when they're missing.

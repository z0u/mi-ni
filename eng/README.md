# Engineering notes

The durable rationale behind `mini`'s storage, artifact, and publishing internals — the *why* that isn't obvious from the code, written for someone (maybe you) returning to this repo cold. The feasibility studies and migration logs that used to live here are gone; this is the distilled conclusion. The skill (`.agents/skills/mi-ni/`) is the *how*; these notes are the *why*.

Start with whichever question you're holding:

- [Artifacts and the content-addressed store](./artifacts.md) — why a step returns an `Artifact` handle instead of a volume `Path`, the CAS-plus-refs layering, and how storage is scoped (the store is project-wide; the memo store and volume are per-experiment).
- [The storage backend: Hugging Face buckets](./storage-backend.md) — why a bucket, and the ~2–3s latency floor that shapes the whole batch-or-parallelize API.
- [Publishing reports to the web](./publishing.md) — why `publish` is a separate, outward-facing verb from `put`, and how a report bundle reaches the web through a single `<base>` switch.
- [Environments: production, dev, and the backup](./environments.md) — why a storage profile replaces the pair and inherits the rest, why dev pins never enter `publish.lock`, and why the backup is a pull from a trust domain the development tokens cannot reach.
- [Reclaiming storage: `mini gc`](./gc.md) — the three durable planes every experiment leaves behind, and the mark-and-sweep that reclaims the CAS without ever collecting a live memo hit.
- [Reproducible GPU runs](./determinism.md) — why the same seed gave three different models, what the `XLA_FLAGS` we set buy and cost, why the flags live in the environment rather than in Python, why a JAX bump moves the digest without moving the memo key (and how a run that straddles one is detected), and why "just tolerate small differences" doesn't survive contact with a content-addressed DAG.
- [Non-goals and recorded decisions](./decisions.md) — chunked datatrees, checkpoints, `obstore`, the HF cache tier, why no hosted experiment tracker, why the code fingerprint is ours rather than a library's, and the open/deferred list with issue links.
- [Operational constraints](./operations.md) — the egress allow-list, Modal gRPC TLS, CORS/Range, which progress transport each execution path uses, and why containers are left unpinned: environment facts that cause confusing failures when they're missing.

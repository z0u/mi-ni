# The storage backend: Hugging Face buckets

*Part of the [engineering notes](./README.md).*

A bucket is a Xet-backed repo type, **mutable, with no git history**: you overwrite in place, and there are no commits to conflict. That last property is load-bearing — independent workers can write results concurrently without a coordination step, where a git-backed dataset repo would 412 on the shared parent. Immutability of `cas/<sha>` is therefore *our* discipline (write-once-by-hash), not the backend's guarantee.

**The latency floor shapes the whole API.** Every bucket read or write pays a fixed ~2–3s round trip before any bytes move (the Xet handshake + commit), independent of size; throughput above the floor is ~10–35 MB/s. Consequences:

- **Batch or parallelize writes.** Eight serial commits took ~15s; one batched commit, or eight *concurrent* commits, ~2–2.6s. The floor is overlappable latency, not a server-side lock — so workers write independently and we batch when files are already in hand.
- **Batch reads the same way.** The floor is per *call*, not per byte: eight sequential cold ref-reads measured ~12s and eight sequential blob gets ~11s, where one batched paths-info + one `download_bucket_files` for the same set is ~3s and ~2s. So `get_refs`/`get_many` resolve a set in one request, a tree `get` prefetches all its children in one pull, and report/eval loops should reach for the batch verbs rather than looping `get_ref`/`get`. (Warm-cache re-reads are local and effectively free — the cache was never the bottleneck, the per-call floor was.)
- **Bucket transfers already ride the Xet layers.** Since `huggingface_hub` ≥ 1.19, `download_bucket_files`/`batch_bucket_files` go through the shared `hf_xet` session, so chunk-level dedup and the `HF_HOME/xet` chunk cache apply to bucket blobs too (on Modal that cache sits on the shared `mini-hf-cache` Volume — see [decisions](./decisions.md)). There's no separate faster path to route reads through; what costs is the per-call metadata round trips, hence the batching above.
- **Keep checkpoints on the volume, not the CAS** (see [Non-goals](./decisions.md)) — content-addressing mutable, superseded state pays the floor repeatedly for nothing.

The bucket holds the durable tier — `cas/<sha>` + `refs/` — and can be **private**: the public, versioned *publish* tier (`published/`, `exports/`) splits off onto a dataset repo when `publish-repo` is set, so persisting an artifact never makes its bytes world-readable. See [Publishing](./publishing.md) for that split ([mi-ni#38](https://github.com/z0u/mi-ni/issues/38)).

# The storage backend: Hugging Face buckets

*Part of the [engineering notes](./README.md).*

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
- **Keep checkpoints on the volume, not the CAS** (see
  [Non-goals](./decisions.md)) — content-addressing mutable, superseded state pays the
  floor repeatedly for nothing.

The bucket holds the durable tier — `cas/<sha>` + `refs/` — and can be **private**: the
public, versioned *publish* tier (`published/`, `exports/`) splits off onto a dataset repo
when `publish-repo` is set, so persisting an artifact never makes its bytes world-readable.
See [Publishing](./publishing.md) for that split (#38).

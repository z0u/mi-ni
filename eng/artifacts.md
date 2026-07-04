# Artifacts and the content-addressed store

*Part of the [engineering notes](./README.md).*

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
partial dedup matters; a single file is fine otherwise. (But see the
[chunked-data non-goal](./decisions.md) — a tree is for a handful of shards, not
thousands of tiny chunks.)

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

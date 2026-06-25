# Content-addressed artifacts: a design sketch

This builds on the [bucket benchmark](./hf-buckets.md) and the handles-not-paths
frame from our threads. It's a sketch, not a spec: I want to show how a
content-addressed `Store` slots into the *current* `memo` / `Ctx` / `Volume`
shapes, and flag the places where the real code pushes back on the tidy version
of the idea. Names and signatures are illustrative.

## Where this slots in

Four things in the current code make this a completion rather than a rewrite.

The memo key is already content-addressed and experiment-name-independent:
`fingerprint(fn, args, version)` in `memo.py` hashes the function's source and
canonicalized args, so identical prep in two experiments produces the same key.
The asymmetry the frame names is real: we hash *inputs* into a key but store
*outputs* as paths. A result is `cloudpickle.dumps(result)` at
`data_dir/_memo/<key>/result.pkl` (`MemoStore.result_dir`), so a returned `Path`
pickles a location into a volume that may have evaporated.

The transactional ordering we'd need mostly exists. `_taskworker.execute_task`
writes `result.pkl`, calls `commit()`, then flips the record to `DONE` — its
docstring already promises that "a poller never sees a settled state whose
artifact hasn't been committed yet." We extend that promise from "the volume was
flushed" to "the referenced blobs are durable."

Steps reach ambient services through context variables, not arguments:
`get_data_dir()` reads a `contextvars.ContextVar` set by `data_dir_context`. A
`Store` handle can ride the same pattern, which matters because of the process
boundary below.

And the `Volume` is already the warm, local thing (`path`, `upload`, `download`).
In this design it stops being the durable home and becomes a checkout cache.

## The handle

```python
@dataclass(frozen=True)
class Artifact:
    sha256: str
    size: int
    name: str                       # logical name, carries the extension
    media_type: str | None = None   # for serving; inferred from name if None
    kind: Literal['file', 'tree'] = 'file'
    children: tuple[Artifact, ...] = ()  # for kind='tree': a manifest
```

A handle is small, JSON-canonicalizable, and location-free. That last property is
load-bearing twice over: it's what lets a result pickle durably, and it's what
lets a *downstream* step's fingerprint stay stable. Today, passing a prep output
into `train` as a path would (if paths were deterministic, which they aren't)
fingerprint `train` by location; passing an `Artifact` fingerprints it by
content. Content-addressing outputs stabilizes the keys of everything that
consumes them.

The `tree` kind is the answer to "few big files or many small ones," which in
this repo is both: the shared tokenized corpus is a few medium files, but
mech-interp activation caches are many small ones with random per-shard access. A
tree is a manifest of child handles — each child its own CAS blob — so you get
per-file dedup and can resolve one shard without pulling the set. The benchmark
is what makes this practical: per-file commits parallelize (8 concurrent in 2.6s
vs 15.4s serial), so a manifest's many small puts aren't the latency trap they'd
be if commits serialized. I'd reach for `tree` over tarring a directory whenever
random access or partial dedup matters; tar only when a directory is always read
whole.

## The store

```python
class Store(Protocol):
    def has(self, sha256: str) -> bool: ...
    def put(self, data: bytes | Path, *, name: str) -> Artifact: ...
    def get(self, art: Artifact, dest: Path) -> Path: ...      # materialize
    def publish(self, art: Artifact, path: str) -> str: ...    # named view -> URL
```

Swappable the way the apparatus is: `LocalStore` (a `cas/<sha256>` tree on disk,
the boring default), `HFBucketStore` (the same layout in a bucket), an `R2Store`
later. Handles don't change when the backend does. `put` is idempotent: hash
first, and if `has(sha)` skip the upload — which makes re-runs and cross-step
duplicates free, and is why uploading before committing the memo costs nothing on
the second pass.

`HFBucketStore` writes blobs to `cas/<sha256>` with `batch_bucket_files`, and
gets Xet chunk-level dedup *underneath* the logical CAS for nothing — two
checkpoints sharing frozen layers store once. Concurrent writes don't conflict,
because buckets are mutable (a git-backed Datasets repo would 412 here). The cost
is that immutability is ours to enforce: write-once-by-hash is a discipline, not
a guarantee from the backend.

## put and get, and the process boundary

The frame wrote `ctx.put` / `ctx.get`, but `Ctx` lives in the *driver* (it runs
`main(ctx)`, suspends on `Pending`, and never executes a step's body). The step
runs in a separate worker — local subprocess or Modal function. So put/get called
*inside a step* can't be methods on that `Ctx`. They follow the `get_data_dir`
pattern instead:

```python
# inside a step body
from mini.store import put, get

def prepare_corpus(cfg) -> Artifact:
    out = get_data_dir() / 'corpus'
    tokenize(cfg, into=out)
    return put(out, name='corpus.bin')   # hashed + uploaded before fn returns
```

`put`/`get` read a `store` context var that the worker enters alongside the data
dir, so the wiring is one line in `execute_task`:

```python
with data_dir_context(store.data_dir), store_context(app.store), progress_context(...):
    result = fn(*args)
(result_dir / 'result.pkl').write_bytes(cloudpickle.dumps(result))  # holds handles
commit(); store.update(key, state=RunState.DONE)                    # unchanged
```

Because `put` uploads synchronously inside `fn`, by the time `result.pkl` is
written the blobs are already durable; the existing write → commit → DONE order
then gives the invariant for free: if a memo says DONE, its bytes resolve. The
driver side does need a resolve too — when `main` passes a returned `Artifact`
into the next step, and when a report reads results back — so `get` is also a
plain function backed by the same store, usable outside a worker.

## Lazy resolve, concurrently

`get` checks the warm volume for `cas/<sha>` first and pulls from the store only
on a miss, materializing into the checkout cache. The benchmark's per-op floor
(~2-3s) means resolving a tree's shards one at a time would serialize painfully,
so the tree resolve fans out:

```python
def get_tree(t: Artifact, dest: Path) -> Path:
    with ThreadPoolExecutor() as ex:
        list(ex.map(lambda c: get(c, dest / c.name), t.children))
    return dest
```

This is the only place `hf://`/fsspec belongs — on the resolve path, where
fsspec's faster small-file reads (0.3s vs 2.7s for 1 KB) actually help, not as
the filesystem itself.

## Publishing to the web

This is what the benchmark changed. A CAS blob at `cas/<sha>` serves as
`application/octet-stream` — no extension, so a browser won't render it. The
named view fixes that: `publish` does a server-side copy *by hash*
(`batch_bucket_files(copy=...)`, ~instant, no bytes moved) to an extensioned
path, and the bucket's resolve URL then sets `Content-Type` from that extension.

```python
url = store.publish(fig, f'reports/{exp}/loss.png')
# -> https://huggingface.co/buckets/<ns>/<bucket>/resolve/reports/<exp>/loss.png
#    served as image/png, Content-Disposition: inline
```

So one bucket is both the durable store and the asset host, and the report's
heavy figures move out of Git LFS. I'd split by size at this boundary: KB-scale
curves stay committed in Git (visible and diffable in the GitHub UI), anything
large is published and referenced by URL. That clears all four LFS problems at
once — writable from Claude Code Cloud, no 1 GB ceiling, shareable mid-flight,
and the report source still readable on GitHub. Keep Pages as the front door;
just have it pull large assets from the bucket.

The caveat to design around: direct serving wants a public bucket. Keep `publish`
separate from `put` and explicitly public-scoped, so the durable CAS can be
private while only deliberately published views are world-readable. Don't let
public exposure become a side effect of persisting a result.

## Cross-experiment reuse needs one more thing

Handles are necessary but not sufficient for the "corpus prepared three times"
problem. They fix the artifact half: outputs land in a global CAS keyed by sha,
so the bytes dedupe across experiments. But that only dedupes *storage* — B still
recomputes the corpus to discover the sha, because the *memo lookup* is
per-experiment. `MemoStore` is scoped by `data_dir` (local `.mini/<name>`, Modal
volume `<name>`), so A's record for key `prepare-abc123` is invisible to B even
though the key is identical.

To skip the *compute*, the record that maps key → result (now a handle) has to be
findable across experiments. The cleanest version, given the fingerprint is
already name-independent: resolve DONE results through a shared,
content-addressed result store keyed by fingerprint, with the per-experiment
`MemoStore` demoted to a run's control plane (progress, heartbeats, the in-flight
set). A step checks the shared store first; a hit anywhere means skip-and-resolve.
I'd make this opt-in per step at first (`ctx.run(prepare, shared=True)`), because
a global memo namespace raises trust questions — one experiment's bug shouldn't
silently poison another's cache — that want more thought than the storage layer
does.

## What I'd build first

1. `Artifact`, `LocalStore`, and contextvar `put`/`get`; `result.pkl` holds
   handles. This alone kills the dangling-path bug within an experiment, with no
   backend change and no network.
2. `HFBucketStore` with the `cas/<sha>` layout and concurrent puts. Durable,
   swappable, cross-process — the thing that makes an agent picking up work in
   Claude Code Cloud actually safe.
3. `publish` for reports and figures; move large assets off LFS, split small/large
   at the Git boundary.
4. Shared-scope memo resolution for cross-experiment compute reuse, once we've
   decided how much to trust a shared namespace.

## Open questions

The trust model for shared memoization is the big one: opt-in per step versus
global, and how a poisoned or stale entry gets invalidated across experiments
that didn't produce it. Garbage collection is the next — a CAS only grows, and
buckets bill per TB, so we need refcounting from live memos plus a sweep
(mutability at least makes the delete cheap). Smaller: whether `Store` should be
async to match `Volume`, or stay sync inside steps and lean on `hf_xet`'s internal
parallelism; and the exact on-disk manifest format for `tree` artifacts.

If this shape looks right, the obvious next step is a real `Store` protocol plus
`LocalStore` and the `put`/`get` contextvar wiring — step 1 above — which is
self-contained and testable without any bucket at all.

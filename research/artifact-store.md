# Content-addressed artifacts: a design sketch

By Claude Opus 4.8

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
    def put(self, data: bytes | Path, *, name: str) -> Artifact: ...   # immutable blob
    def get(self, art: Artifact, dest: Path) -> Path: ...             # materialize
    def publish(self, art: Artifact, path: str) -> str: ...           # named view -> URL
    def set_ref(self, name: str, art: Artifact) -> None: ...          # mutable name -> sha
    def get_ref(self, name: str) -> Artifact | None: ...
```

The `cas/<sha>` blobs are immutable; `set_ref`/`get_ref` are a small mutable layer
of names over them (the git objects-and-refs split). One ref layer covers three
needs that recur below: checkpoint pointers, the cross-experiment by-name lookup,
and `publish`'s named views.

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

This reopens a question worth getting right. The publish tier could live in a
Hugging Face *dataset repo* or in a bucket, and one tempting case for a repo rests
on the belief that a bucket exposes no public, CDN-backed file URL. The benchmark
contradicts that belief: a named view over `cas/<sha>` serves with the correct
`Content-Type` from a public bucket, so that particular reason for a repo falls
away. What's left is narrower and real. A repo keeps git history and lets you pin a
citation to `/resolve/<commit-sha>/`. A bucket is mutable with no history, but
serving a content-addressed path (`cas/<sha>` through a named view) gives immutable
*content* without git, so the residual trade is permanence and the dataset-viewer
ecosystem against running one backend instead of two. Rather than commit now, I'd
keep `HFStore` backend-swappable (repo or bucket), so the publish tier and the
deferred working store can later collapse onto a single bucket instead of splitting
across the two.

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

There are two ways to scope that shared lookup, and I've come around to the broader
one. The cautious version is opt-in per step (a `shared=True` flag), out of worry
about a global namespace. The broader version scopes the store to the whole
*project* and tags each record by experiment. Project scoping answers the worry:
one project is one trust domain, so cross-experiment cache poisoning is just your
own bug to fix, and the opt-in caution really belongs at the cross-*project*
boundary instead. The cost is genuine, and it sits in the control plane rather than
the CAS: `cancel`, `retry`, and budget teardown have to become tag-scoped, or a
single `mini status` on an over-budget experiment cancels every experiment's
in-flight tasks. Worth solving alongside the storage change, since they land
together.

## Checkpoints are a different category

Mid-task checkpoints — the periodic state a long step writes so it can resume
after a hard crash — are not step outputs, and shouldn't enter the handle graph.
They're mutable and superseded on the next write, so content-addressing them pays
the per-op floor repeatedly and churns the CAS with blobs obsolete minutes later.
The giveaway is that resume needs to find "the latest checkpoint for this step" by
a stable *name*, not by a hash you only learn after writing it.

So checkpoints want the volume, not the CAS — with one correction. mini commits
the Modal volume once at the step boundary (`execute_task` is called with
`commit=volume.commit`), not mid-task, so a hard kill can lose everything written
since the last background flush. If a checkpoint is expensive to recompute, call
`volume.commit()` explicitly right after writing it rather than trusting
auto-commit timing. Volume-based resume then works whenever the same durable named
volume is re-mounted, which covers crash-and-retry in the same project on the same
backend (and, under a project-scoped volume, across experiments in the
project).

It breaks in exactly the case this whole design exists for: a cross-process
handoff where the volume isn't shared — close the laptop, an agent resumes in the
Cloud, or local hands off to Modal. There the checkpoint has to live in the
durable store. That points at a gap in the sketch above: the store wants a small
*mutable ref layer* (`name → sha`) over the immutable `cas/<sha>` blobs — the git
objects-and-refs split. One ref layer serves three needs: a `refs/checkpoints/<step>`
pointer that resume overwrites, the named `publish` views, and the
cross-experiment by-name lookup. A checkpoint becomes an explicit
`checkpoint(path, name)` that updates a ref, distinct from the immutable `put`.
Xet's chunk dedup makes each upload incremental rather than a full copy, but the
latency floor still says: keep the cadence coarse, and default to the volume.

## What I'd build first

1. `Artifact`, `LocalStore`, and contextvar `put`/`get`; `result.pkl` holds
   handles. This alone kills the dangling-path bug within an experiment, with no
   backend change and no network.
2. `HFBucketStore` with the `cas/<sha>` layout and concurrent puts. Durable,
   swappable, cross-process — the thing that makes an agent picking up work in
   Claude Code Cloud actually safe.
3. `publish` for reports and figures; move large assets off LFS, split small/large
   at the Git boundary.
4. Project-scoped memo resolution for cross-experiment compute reuse, with the
   control-plane operations tag-scoped so budget teardown can't cancel a sibling
   experiment's tasks.

## Open questions

The trust model for shared memoization is the big one. Project scoping narrows it:
the question then moves to the cross-*project* boundary, where a poisoned or stale
entry would have to be invalidated across projects that didn't produce it. Garbage
collection is next. A CAS only grows, and buckets bill per TB, so we need
refcounting from live memos plus a sweep. The backend matters here: a bucket has
server-side `rm`, while a Modal `Dict` (the control plane) has only `clear()`, no
per-key delete, which shapes what tag-scoped cleanup can do. Smaller: whether
`Store` should be async to match `Volume`, or stay sync inside steps and lean on
`hf_xet`'s internal parallelism; and the exact on-disk manifest format for `tree`
artifacts.

If this shape looks right, the obvious next step is a real `Store` protocol plus
`LocalStore` and the `put`/`get` contextvar wiring, step 1 above, which is
self-contained and testable without any bucket at all.

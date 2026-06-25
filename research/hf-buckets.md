# Hugging Face buckets as experiment storage

By Claude Opus 4.8

A quick feasibility study, not a decision. I measured I/O latency for Hugging
Face buckets and compared them to a Modal Volume, to see whether a
`HFBucketVolume` could sit behind an `Apparatus`. The numbers look workable, but
how we'd integrate it is still open. Treat everything here as provisional.

## What a bucket is

A bucket is its own repo type on the Hub, addressed as
`hf://buckets/<namespace>/<name>`, distinct from dataset and model repos. It is
Xet-backed, mutable, and has no git history: you overwrite in place and there
are no commits to rebase or conflict. The Python API exposes it through
`HfApi.batch_bucket_files`, `download_bucket_files`, and `sync_bucket`; the
`hf` CLI mirrors these as `hf buckets ...` and `hf sync`. The same content is
also reachable over `fsspec` via `HfFileSystem` (`hf://buckets/...`).

I ran the tests against the public bucket `z0u/mi-ni-store`. The token we use is
fine-grained and scoped to that one bucket, which is a nice property: a leaked
token can't touch the rest of the namespace.

## The shape of the latency

Every bucket read or write pays a fixed cost of roughly two to three seconds
before any bytes move. That floor is request latency, not transfer: the Xet
handshake, content-addressing negotiation, and the commit each cost a round
trip. Above the floor, throughput sat around 8–20 MB/s on writes and 10–35 MB/s
on reads, with enough run-to-run variance (Xet warm-up, I think) that I wouldn't
quote a single figure.

| Payload | From this container (HfApi) | From a Modal worker |
| --- | --- | --- |
| 1 KB | write 1.8s / read 2.7s | write 2.4s / read 2.2s |
| 1 MB | write 2.1s / read 2.1s | write 3.3s / read 3.0s |
| 16 MB | write 2.9s / read 2.3s | write 4.8s / read 3.9s |
| 64 MB | write 7.3s / read 3.9s | write 8.5s / read 6.5s |

Reading through `HfFileSystem` was consistently faster than the bucket download
API for small files (around 0.3s versus 2.7s for 1 KB), probably because it
skips the temp-file plumbing and streams directly. For writes the two were
comparable.

## Buckets versus a Modal Volume

The contrast is stark, and it's the crux of any design. A Modal Volume is a
local mount inside the function: reads and writes hit local disk and cost
microseconds to milliseconds. You pay only at the boundaries, when you `commit()`
to publish or `reload()` to see other workers' writes. A bucket charges the
two-to-three-second round trip on every operation.

| Operation | Modal Volume | HF bucket |
| --- | --- | --- |
| In-function read/write (16 MB) | ~0.02–0.03s | ~3–5s |
| Publish / commit (16 MB) | commit ~6.3s | write ~3–5s |
| See remote changes | reload ~0.2–0.5s | read ~2–4s |

So for data touched repeatedly inside a worker, the Volume wins by orders of
magnitude. For a one-shot publish or fetch, the two are roughly even: a bucket
write (~3s) is in the same range as a Volume commit (~2.3s fixed, more with
size). Where the bucket pulls ahead is reach: it has a public URL and lives
outside Modal, so it can serve data to a browser, a notebook, or another
machine. A Modal Volume can't.

## Batching, or concurrency

Naive per-file writes are slow because each commit pays the full floor: eight
separate commits ran in 15.4s, about 1.7s each. The obvious fix is to batch many
files into one commit (eight files in 1.97s). The less obvious fix is to keep
the separate commits but issue them concurrently, which I tested: eight
concurrent commits finished in 2.59s, nearly matching the single batched commit.

That tells me the floor is latency we can overlap, not a server-side lock. It
works because buckets are mutable: concurrent commits don't fight over a shared
parent the way they would on a git-backed repo, where you'd hit conflicts and be
forced to batch lower down. So workers can write results independently, without a
coordination step. Batching is still tidier and atomic when the files are already
in hand, so I'd reach for it first and use concurrency for results that arrive
piecemeal.

For bulk directory moves, `hf sync` does the sensible thing. Pushing 101 files
(17 MB) took 5.1s, a no-op re-sync took 1.1s (it diffs on size and mtime), and a
fresh download took 5.7s. That's the natural fit for syncing a Volume to or from
a bucket.

## A possible HFBucketVolume

If we add this as a `Volume`, the interface fits: `path` for the local working
directory, and `upload` / `download` mapping onto `batch_bucket_files`,
`sync_bucket`, or `HfFileSystem`. The interesting question is caching.

The expensive thing is the round trip, so a cache that avoids re-fetching is the
real win. My current guess is that the cache should be pluggable rather than a
Modal Volume hard-wired behind every bucket. Within a single function, ephemeral
local disk (or `HF_HOME`, where the Xet client already caches chunks) reads just
as fast as a Volume would, and avoids the Volume's own commit and reload
overhead. A Modal Volume earns its place only when many workers or runs would
otherwise cold-download the same data: point the cache at a shared Volume and
they share one warm copy. So I'd sketch it as `HFBucketVolume(cache=...)`,
defaulting to local disk, with a Modal Volume as an opt-in warm cache. The bucket
stays the source of truth; the cache is just an accelerator.

There's a third option I tested but would not lead with: mounting the bucket as
a filesystem with [hf-mount](https://github.com/huggingface/hf-mount). It works,
and feels fast (a 1 KB write's `close()` returned in 0.8s, a read in 0.25s),
because writes buffer locally and flush asynchronously. That asynchrony is also
the catch: writes are only eventually durable (a two-second flush debounce),
cross-client visibility lags by about ten seconds, and it needs `/dev/fuse` plus
`CAP_SYS_ADMIN`. I got it running here as root; I doubt it runs inside an ordinary
Modal function. A neat option for interactive exploration, less so for checkpoint
correctness.

## Serving figures

A bucket can serve images with the right content type, which makes it appealing
for published notebook assets. A resolve URL like
`https://huggingface.co/buckets/z0u/mi-ni-store/resolve/figs/plot.png` returns a
302 to a signed CDN URL that carries `response-content-type` set from the file
extension, with `Content-Disposition: inline`. I confirmed the mapping across
formats: `.png` to `image/png`, `.svg` to `image/svg+xml`, `.html` to
`text/html`, and an extensionless `.bin` falling back to
`application/octet-stream`. So `<img src=".../resolve/figs/plot.png">` renders
inline. The type is inferred from the extension, so files need honest names, and
there's a redirect hop that matters little for display.

## On obstore

I looked at whether [obstore](https://github.com/developmentseed/obstore) buys us
anything, since it's faster than fsspec for the stores it supports. It doesn't
support Hugging Face: only S3, GCS, and Azure. The only way to bridge is through
fsspec, which forfeits the native-Rust speed that makes obstore worth using. And
we don't lose much by skipping it, because `hf_xet` already does parallel chunked
transfer natively under `HfApi`. obstore would only matter if mi-ni also targeted
S3, GCS, or Azure directly.

## Open questions

A few things to settle before committing to this:

- Where the cache should live by default, and whether the Modal-Volume-as-cache
  path is worth the extra moving parts.
- Whether `upload` / `download` should prefer `sync_bucket` (rsync-like, good for
  directories) or `batch_bucket_files` (atomic, good for known file sets), and
  how that interacts with memoized results.
- How durability expectations map onto the bucket's commit model, especially if
  we ever consider the hf-mount path.

## Reproducing

Two environment details cost me time and are worth recording. Bucket I/O needs
`*.xethub.hf.co` on the egress allow-list; without it, metadata calls to
`huggingface.co` succeed while every byte transfer hangs on a 403. Serving
figures additionally needs the CDN host (`*.cdn.hf.co`). And Modal's gRPC client
builds its TLS context from certifi alone, so behind a TLS-inspecting proxy it
needs the system CA folded in; `mini._tls.ensure_grpc_trusts_system_ca` already
does this.

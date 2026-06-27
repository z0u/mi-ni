# Artifacts and the content-addressed store

A memo result is the *small* thing — a dict of metrics, a handle. The *large*
bytes a step produces (an activation cache, an eval dump, a figure) belong in the
**artifact store**, not in the result and not as a bare volume `Path`.

Returning a `Path` pickles a *location* into the result, and that location lives
in a volume that may have evaporated by the time another process, another
experiment, or a report reads the result back. Instead, a step `put`s its bytes
and returns an `Artifact` — a small, location-free handle (a sha, a size, a
name).

```python
from mini import get_data_dir
from mini.store import put, get, get_ref, set_ref, publish

def extract(cfg) -> dict:
    cache = get_data_dir() / 'acts'
    run_model(cfg, into=cache)
    art = put(cache, name='activations')   # hashed into the store; handle returned
    return {'cfg': cfg.id, 'activations': art}
```

`put`/`get` resolve an **ambient store** the worker enters around the step — the
same pattern as `get_data_dir()`. They work inside any step with no plumbing;
outside a step (a notebook/report), get the store from the apparatus:
`store = LocalApparatus(NAME).store()` and call `store.get(...)` directly.

## Why a handle, not a path

- **Durable results.** A handle carries no location, so the result pickles
  durably and resolves from anywhere that can reach the store.
- **Stable downstream keys.** Passing a `Path` into the next step fingerprints it
  by location; passing an `Artifact` fingerprints it by *content*, so a
  consumer's memo key only moves when the bytes actually change.
- **Dedup for free.** Blobs are keyed by content (`cas/<sha256>`), so identical
  bytes coincide and `put` is idempotent (hash first, skip if present).

## Files and trees

`put(bytes | Path, name=...)`. A directory becomes a **tree** artifact: each file
is its own blob (so a directory of many small shards dedups per-file and resolves
one shard without pulling the set), and the handle carries the manifest. `name`
is the logical name — carry the extension; it sets the served media type.

`get(art, dest)` materializes a file to `dest`, or a tree into the directory
`dest` (children resolve concurrently). Reach for a tree when random access or
partial dedup matters; otherwise a single file is fine.

## The store is project-scoped (sharing across experiments)

Unlike the memo store and volume (one per experiment), the artifact store is
**one per project** — it sits a `store/` beside the experiment volumes. So an
artifact one experiment produces is visible to every experiment in the project,
content-addressed.

A small mutable **ref** layer names views over the immutable blobs (the git
objects-and-refs split). That's how one experiment hands an asset to another by a
stable name, without the consumer knowing the producer's memo key:

```python
# producer experiment
set_ref(f'activations/{dataset}', art)

# consumer experiment — no recompute, no shared volume
art = get_ref(f'activations/{dataset}')
local = get(art, get_data_dir() / 'acts-in')
```

See `docs/acts` (producer) and `docs/probe` (consumer) for a runnable pair.

## Choosing the backend (local vs. Hugging Face bucket)

The backend is configuration, not code — nothing in an experiment or report
changes:

- **No bucket configured** → `LocalStore`, a `cas/<ab>/<sha>` tree under
  `.mini/store`. The default; no network. Project-wide sharing works *locally*.
- **A bucket configured** → `HFStore`, the same layout over a Xet-backed Hugging
  Face bucket, shared across *machines and backends*: a Modal worker `put`s a
  blob; a local report or another experiment `get`s it back, no shared Volume.
  The local dir demotes to a warm cache (`.mini/store-cache/hf`).

Set the bucket once in `pyproject.toml` so it travels with the repo (set in one
place, not three):

```toml
[tool.mini]
store-bucket = "your-namespace/your-bucket"
```

`MINI_STORE_BUCKET` overrides it for a one-off shell or CI. `mini run --app modal`
forwards the resolved bucket + token into the worker via a Modal Secret. Bucket
I/O needs `*.xethub.hf.co` (and `*.cdn.hf.co` for serving) on the egress
allow-list.

Auth: `./go auth` logs into Hugging Face (a fine-grained token with read+write to
the bucket). `hf` caches it; the store and the Modal Secret read it from there, so
`HF_TOKEN` need not be exported — the bucket name isn't a secret, only the token.

## Publishing to the web

`publish(art, path)` is the store's outward-facing verb: it exposes a blob at a
named, extensioned path and returns a URL (the extension drives the served
`Content-Type`). It's deliberately separate from `put`, so persisting a result
never publishes it as a side effect.

```python
url = store.publish(art, f'reports/{exp}/loss.png')
```

`LocalStore` returns a `file://` URL (the published view lives under the project
store). With `MINI_STORE_BUCKET` set, `HFStore` returns a real `https://` resolve
URL for the same handle — a server-side copy *by xet hash* (no bytes moved) to an
extensioned path, served with a `Content-Type` from that extension. The bucket is
**public**, so `publish` is the deliberate outward step; the durable CAS only goes
public because we share one bucket for now (a private store + public publish bucket
is a later split — see `todo.md`).

**Reports don't call `publish` directly** — they go through a report bundle.

## Report bundles

A report is a **bundle**: one Marimo HTML document plus its heavy assets (figures,
data blobs). `mini.reports` owns both halves — producing the assets and repointing
them at the bucket on publish.

**Produce.** Set a `Publisher` once in the report's setup cell; every `themed` figure
then externalizes through it (figure cells are unchanged), and `asset_url` is the
general verb for any blob a report's JS reads (a large JSON for a data browser, an
SPA's data files):

```py
from mini.reports import use_publisher, report_bundle

pub = use_publisher(report_bundle(__file__))   # assets → this report's __marimo__/_assets/
url = pub.asset_url(points_json, name='points.json')   # -> '_assets/<sha>/points.json'
```

Each asset is written to `_assets/<sha>/<name>`: the SHA directory is the content
address (identical bytes are written once and shared across reports), and the readable
`name` is the leaf — so a browser "Save as" suggests that name (it takes the URL's last
segment; the bucket sets no `Content-Disposition`). With no publisher, figures inline
as self-contained `data:` URIs, so a no-frills export still works.

**Publish.** The relative URL is the point: the *same* HTML works two ways. Opened
locally, `_assets/…` resolves to the co-located files (offline; the figures are real
PNGs). Published, `scripts/build_site.py` uploads `_assets/` to the bucket and inserts
one `<base href>` in the `<head>` so the same relative URLs resolve there — no per-URL
rewriting.

Because `<base>` repoints *every* relative URL, the rule is **the only relative URLs in
a report are its assets**. Author-written nav/source links would break against the
bucket, so `build_site` resolves them: a link to another report or `.md` becomes its
rendered page, a link to a source file becomes its GitHub source, and anything it can't
place is left alone with a warning. Write natural relative links
(`[experiment](./experiment.py)`); the absolute targets are derived from the git remote
(override with `MINI_SITE_URL` / `MINI_SOURCE_URL`). Design notes:
`research/reports-hfstore-migration.md`.

## Checkpoints are different

Mid-step checkpoints (periodic state for crash-resume) are *not* step outputs:
they're mutable and superseded, and resume finds "the latest for this step" by a
stable name, not a content hash. Keep those on the volume (`get_data_dir()`), not
in the CAS.

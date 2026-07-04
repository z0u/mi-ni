# Handoff: issue #15 — GC for run records, Modal state, and the artifact CAS

Status: **implementation complete and green; tests for the new surface + docs + issue comment still to do.**
Delete this file before merge.

## What's in this commit

Three legs, all working:

1. **Forward artifact index** — `_taskworker` now writes a
   `result-<gen>.artifacts.json` sidecar (sorted blob shas from the result,
   found by `artifact_shas()` in `store.py`, an object-graph walk pruned at
   `_OPAQUE` types). The CAS mark phase reads these tiny files instead of
   unpickling results; unpickle is the fallback for legacy records only.
   This is the "fast when the store is huge" answer.
2. **Backend-generic per-experiment gc** — `GcIO` adapter in `gc.py`
   (`LocalGcIO` over the filesystem, `ModalGcIO` over a Modal Volume via one
   recursive `listdir`). `Apparatus.gc_io(store)` vends the right one;
   `mini gc <name> --app modal` now works. Modal orphan dirs are the normal
   end state of 7-day Dict expiry — collectible by design.
3. **CAS mark-and-sweep** — `mini gc --store [--grace 14d] [--apply]`.
   Mark = every record (current AND superseded) across all experiments
   (`collect_store_roots`, which peeks Modal Dicts read-only via
   `Dict.from_name` + `hydrate`, volumes with `create=False`) plus every ref
   (`set_ref` is the pin mechanism). Sweep = `plan_store_gc`/`apply_store_gc`
   over `Store.list_blobs()` (implemented on LocalStore and HFStore).

## Safety design (the issue's first question)

Fail-closed mark: `StoreGcError` aborts the whole sweep on any RUNNING/PENDING
task, unreadable result, or unknown `.app` backend stamp. Grace window
(default '14d', git-style) keeps unreferenced blobs younger than the cutoff —
covers `has()`-skip races and colleagues' unseen checkouts. Dry-run default.
Bucket-configured-but-no-token aborts rather than silently sweeping the local
fallback store. `HFStore.delete_blobs` purges the warm cache so `has()` can't
lie afterwards (other machines' caches remain a documented caveat).

## Verified

`ruff check src tests`, `ty check src`, and
`uv run pytest tests -q -x --ignore=tests/mini/test_hf_store.py` → 242 passed.
(test_hf_store.py is the pre-existing network-gated suite, untouched.)

## Remaining work

- **Tests** (`tests/mini/test_store_gc.py`, planned in detail):
  - `artifact_shas` walker: nested containers/dataclasses, tree children
    included, tree-manifest sha never collected (it's not a stored blob).
  - Sidecar e2e: a step that `put`s → sidecar exists, `result_artifacts`
    reads it; stale sidecars swept by the attempt-files leg.
  - LocalStore sweep e2e: unreferenced blob collected at `--grace 0d` only
    after the superseded record is gc'd; referenced/ref-pinned/in-grace kept;
    RUNNING record or corrupt result → `StoreGcError`; legacy no-sidecar
    record marks via unpickle fallback.
  - HFStore against a fake api object (SimpleNamespace entries are enough for
    `list_bucket_tree`; assert 500-per-batch deletes + warm-cache purge).
  - Modal leg via fakes: fake volume with `listdir`/`remove_file` +
    `ModalRecordStore` over a plain dict; ModalGcIO plan/apply incl. orphan
    dir from an expired Dict record.
  - CLI: `--store` dry-run/apply; `name` + `--store` together errors.
  - Latent pre-existing bug while you're there: `test_hf_store.py` teardown
    appends `f'cas/{art.sha256}'` (3×) but the real key is
    `_cas_key(sha)` = `cas/<sha[:2]>/<sha>`, so cleanup misses blobs.
- **Docs**: storage.md section on gc/retention — pin-with-`set_ref` contract,
  grace-window caveats for teams/multiple checkouts; update todo.md index.
- **Issue #15 comment** answering the safety/speed questions; note scope-outs:
  #38 interaction deferred, Dict-expiry memo-hit loss is a separate concern,
  volume-local per-experiment `store/` is reclaimed by deleting the volume.
- Optional: run the network-gated HF integration tests (keys are available)
  and a Modal smoke of `mini gc <exp> --app modal`.

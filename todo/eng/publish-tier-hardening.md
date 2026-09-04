---
status: done
tags: [publishing, storage]
closed: 2026-09-04
bundle: storage-control-plane
---
# Publish-tier hardening — private CAS, public publish bucket

Tracked as [#38](https://github.com/z0u/mi-ni/issues/38) — split the private CAS from the public publish bucket, and make the publish tier citable and versioned via a dataset repo.

Stems from the same list as [`eng/decisions.md`](/eng/decisions.md). Only matters once the template is used for work that shouldn't be world-readable by default.

The rest of that list is settled. [#46](https://github.com/z0u/mi-ni/issues/46) shipped (gen-fenced `set_ref`/`publish` + `StaleWriteError`, [PR #56](https://github.com/z0u/mi-ni/pull/56)). [#37](https://github.com/z0u/mi-ni/issues/37) (implicit cross-experiment dedup + shared working volume) closed as not planned — the explicit ref path covers reuse; reopen only if identical-prep recompute becomes a real recurring cost. [#15](https://github.com/z0u/mi-ni/issues/15), GC across the control plane, I/O-plane volume dirs, and the CAS, shipped in two cuts (`mini gc <name>`, then `mini gc --store`) with the rationale and safety posture in [`eng/gc.md`](/eng/gc.md). This item is the only thing left that would reshape the CAS leg of that sweep.

## Notes

**2026-09-04, Fable** — Done. `publish-repo` is set for this repo (`z0u/mi-ni-pub`, public, git-backed), `publish.lock` pins every report, and the site build reads the dataset repo alone. The bucket `z0u/mi-ni-store` is public on purpose *here*, because this repo is the template; projects built from it keep their bucket private, which is what the `bucket_publish` skip in `tests/mini/test_hf_store.py` assumes. One chore left: the bucket still carries the pre-split prefixes `published/` (27 files), `exports/` (19) and `figs/` (4), about 2.3 MB that nothing links to. `mini gc --store` sweeps `cas/` only; that prune belongs with the orphan-cleanup idea in `exports-go-stale-on-rename.md`, or a one-off delete.

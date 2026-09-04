---
status: partial
tags: [publishing, storage]
bundle: storage-control-plane
---
# Publish-tier hardening — private CAS, public publish bucket

Tracked as [#38](https://github.com/z0u/mi-ni/issues/38) — split the private CAS from the public publish bucket, and make the publish tier citable and versioned via a dataset repo.

Stems from the same list as [`eng/decisions.md`](/eng/decisions.md). Only matters once the template is used for work that shouldn't be world-readable by default.

The rest of that list is settled. [#46](https://github.com/z0u/mi-ni/issues/46) shipped (gen-fenced `set_ref`/`publish` + `StaleWriteError`, [PR #56](https://github.com/z0u/mi-ni/pull/56)). [#37](https://github.com/z0u/mi-ni/issues/37) (implicit cross-experiment dedup + shared working volume) closed as not planned — the explicit ref path covers reuse; reopen only if identical-prep recompute becomes a real recurring cost. [#15](https://github.com/z0u/mi-ni/issues/15), GC across the control plane, I/O-plane volume dirs, and the CAS, shipped in two cuts (`mini gc <name>`, then `mini gc --store`) with the rationale and safety posture in [`eng/gc.md`](/eng/gc.md). This item is the only thing left that would reshape the CAS leg of that sweep.

## Notes

**2026-09-04, Fable** — The code half is done: `publish-repo` is set for this repo (`z0u/mi-ni-pub`, public, git-backed, 49 commits), `publish.lock` pins every report, and the site build reads the dataset repo alone. Two operational steps remain, both under Sandy's login rather than in code. (1) The bucket `z0u/mi-ni-store` is still **public**, so the CAS is not yet private, which is what the title promises and what `tests/mini/test_hf_store.py` (`bucket_publish` skip) assumes once `publish-repo` is set. Flipping it is a settings change and reversible; the build never reads the bucket, so the site should be unaffected. (2) The bucket still carries the pre-split prefixes `published/` (27 files), `exports/` (19) and `figs/` (4), about 2.3 MB of dead bytes that nothing links to (grep found no `buckets/z0u` URLs outside tests). `mini gc --store` sweeps `cas/` only, so these want a one-off delete, or a `--prefixes` extension of the sweep. Once both are done this item is closeable; the non-prod design (`non-prod.md`) also builds on the split.

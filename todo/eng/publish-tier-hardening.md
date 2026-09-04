---
status: open
tags: [publishing, storage]
bundle: storage-control-plane
---
# Publish-tier hardening — private CAS, public publish bucket

Tracked as [#38](https://github.com/z0u/mi-ni/issues/38) — split the private CAS from the public publish bucket, and make the publish tier citable and versioned via a dataset repo.

Stems from the same list as [`eng/decisions.md`](/eng/decisions.md). Only matters once the template is used for work that shouldn't be world-readable by default.

The rest of that list is settled. [#46](https://github.com/z0u/mi-ni/issues/46) shipped (gen-fenced `set_ref`/`publish` + `StaleWriteError`, [PR #56](https://github.com/z0u/mi-ni/pull/56)). [#37](https://github.com/z0u/mi-ni/issues/37) (implicit cross-experiment dedup + shared working volume) closed as not planned — the explicit ref path covers reuse; reopen only if identical-prep recompute becomes a real recurring cost. [#15](https://github.com/z0u/mi-ni/issues/15), GC across the control plane, I/O-plane volume dirs, and the CAS, shipped in two cuts (`mini gc <name>`, then `mini gc --store`) with the rationale and safety posture in [`eng/gc.md`](/eng/gc.md). This item is the only thing left that would reshape the CAS leg of that sweep.

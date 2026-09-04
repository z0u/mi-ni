---
status: open
tags: [publishing, storage]
opened: 2026-08-13
---
# A report's inputs outside its own directory are invisible to the publish check

Split from the publish-check item that fixed the local half (the check only saw changed notebooks): a report's own directory now counts as part of it (`mini.reports.input_dir`). The remaining case is an input that isn't in that directory — a restyle in `src/mini/vis/*.py`, a change to a shared plotting helper — which every report reads and no report contains. Its figures are baked PNGs in the bundle, so a restyle genuinely dates all of them, and nothing local says so.

Two ways to see it, and they differ in what they cost.

The **cheap version** is a git diff over `src/mini/vis/*.py` that flags every report at once. It needs nothing new, but it's a blunt instrument: any touch to a shared module asks for a republish of every report, most of which change no pixel, and a check that over-asks gets labelled `skip-publish-check` and stops being read.

The **sharp version** is the one the original note wanted: compare the store refs the last export resolved — `_assets/provenance.json` records them, see `PROVENANCE_ASSET` — against what those refs point at now. That says which reports actually read data that moved. The obstacle is placement rather than logic: the sidecar rides the *published* bundle, so reading it means fetching from the bucket, and the publish check is deliberately store-free and token-free (`eng/publishing.md`, and the workflow comment in `lint-check.yml` says so out loud). Either it stops being a pure CI check, or the pinned bundle's ref set gets mirrored somewhere local — a manifest beside `publish.lock`, written at publish time, which keeps CI reading nothing but git.

That last option looks like the shape worth costing first: it puts the ref set on the same footing as the pin (identity travels with the code, evidence stays in the store), and it would also give `mini gc` and the orphan-export cleanup a local record of what each published bundle depends on. Not urgent — the directory heuristic covers the common re-run case, and this one is a restyle, which is usually deliberate enough to be remembered.

## Notes

**2026-09-03, backport** — Ported from sca2 with the code that cites it, so some details above describe that project's tree; the reasoning is what the shared code relies on.

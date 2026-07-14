# Todo

Scratchpad for deferred work that isn't worth a tracking issue yet. When something
here grows real, promote it to a GitHub issue and remove it from this list.

Everything below the scratch section is the prioritized index into GitHub
issues. Durable design rationale and recorded decisions live in
[`eng/`](./eng/README.md); each open issue
also carries a grounding comment with current file:line refs, so it should be
readable cold without re-deriving code state.

## Scratch (backported from sca2, 2026-07-14)

- **Publish-tier exports go stale on rename.** `export_key` derives from the
  docs-relative path, so moving a notebook orphans its synced bundle: the build
  looks for the new key, skips with a warning, and the site 404s while
  `index.md` still links the page. Prevention: teach `./go publish` (or the
  build) to list remote export keys and warn on ones with no matching notebook,
  and/or a `./go publish --move old new` verb. Consider folding orphan cleanup
  into `mini gc --store`.

- **PR publishes land on the prod publish tier.** `./go publish` from a PR
  branch writes `exports/<key>/` on the *production* tier — a new report sits
  there dark until main links it (fine; the PR preview even depends on it),
  but re-publishing an *existing* key from a branch silently swaps the assets
  under the live site's stale HTML. If that bites, publish PR exports to a
  `pr-<n>` git revision of the dataset repo (`upload_folder(revision=...)`,
  preview `<base>` at `resolve/pr-<n>/`). See eng/publishing.md.

- Cross-experiment lineage is **auto-detected**: `set_ref` in a task worker
  stamps producer identity onto the ref, `get_ref` records the resolution on
  the task record (`upstream_refs`), and the driver rolls both into
  `lineage.upstreams`. Known gaps: refs written by the interactive `Apparatus`
  (`app.map` in a notebook) or driver-side code are unstamped, and a consumer
  served entirely from memo hits records nothing new — its previously-recorded
  `upstream_refs` persist on the old records, which is usually what you want.
  Pre-existing refs stay unstamped until their publish step re-runs.

- Modal `mem_total_gb` in a task's `env` reads the *host* total from
  `/proc/meminfo` (gvisor shows the whole node), not the container's memory
  limit. Fine as a coarse "what class of machine" signal; if we ever want the
  true per-container cap, read the requested `memory=` from the role config
  instead (or the cgroup limit, if gvisor exposes it).

- `mini.temporal` can't drive feedback control. `DynamicProp.set()` retargets
  mid-flight from the current (value, velocity) state — exactly what a
  controller needs — but experiments consume schedules via `realize_timeline`,
  which bakes the dopesheet into a static per-step array before training, and
  the dopesheet's own keyframes would fight any runtime `set()` calls on the
  same prop. If feedback-driven schedules become standard, consider a Timeline
  mode where a prop is declared "controlled": keyframes set its
  *bounds/defaults* and a callback supplies the live value.

## Backlog, grouped by what a single dev session should bundle

**Quick wins.** All shipped: #39 and #36 (PR #51), #19 (queued ≠ running,
PR #54), #47 (per-experiment backend memory for `--app`).

**Storage/control-plane design.** These stem from the same list in
[`eng/decisions.md`](./eng/decisions.md):

- #38 — publish-tier hardening (private-CAS/public-publish bucket split;
  citable versioned publish via a dataset repo). Only matters once the template
  is used for work that shouldn't be world-readable by default.
- Settled: #46 shipped (gen-fenced `set_ref`/`publish` + `StaleWriteError`,
  PR #56). #37 (implicit cross-experiment dedup + shared working volume) closed
  as not planned — the explicit ref path covers reuse; reopen only if
  identical-prep recompute becomes a real recurring cost.

**Sequence after the above:**

- #15 — GC across the control plane, I/O-plane volume dirs, and the CAS.
  Shipped in two cuts: the local per-experiment control-plane + I/O-plane sweep
  (`mini gc <name>`, PR #49), then the Modal Volume sweep and the CAS
  mark-and-sweep (`mini gc --store`, PR #60). Rationale and safety posture in
  [`eng/gc.md`](./eng/gc.md). Only #38 (bucket split) would
  still reshape the CAS leg; the `mini-hf-cache` Volume (#50) stays out of scope
  (pure cache — `modal volume delete mini-hf-cache` is a safe reset).

**Orthogonal, no code overlap with the above:**

- #45 — docs rework. Touches `docs/`, `README.md`, `eng/`, not `src/mini/`.
  Can run in parallel with anything.
- #57 — CLI DevX: passing a name to `retry`/`run` dies with a raw traceback
  (tick verbs take a file, read verbs a name). Tier 1 (friendly error + help
  text on the `path` positional) is a quick win in `src/mini/__main__.py`.

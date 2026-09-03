# Todo

Scratchpad for deferred work that isn't worth a tracking issue yet. When something
here grows real, promote it to a GitHub issue and remove it from this list.

Everything below the scratch section is the prioritized index into GitHub
issues. Durable design rationale and recorded decisions live in
[`eng/`](./eng/README.md); each open issue
also carries a grounding comment with current file:line refs, so it should be
readable cold without re-deriving code state.

## Scratch (backported from sca2, 2026-09-03)

- **Adopt sca2's `todo/` tree?** sca2 replaced its `todo-*.md` files with one file per item under `todo/{eng,science,style}/`, six-key front matter, indexed by `scripts/todo.py` behind `./go todo` (with `--priority`, `--grep`, and a `--check` lint gate). The shape is worth having in a template; the items aren't. Self-contained: port `todo.py`, the `./go` verb, the lint gate, `test_todo.py`, and migrate this file into `todo/eng/`. Until then, a few ported comments cite sca2's `todo/eng/*.md` by URL.

- **Publish the demo reports once**, with `./go publish --all`, so `docs/publish.lock` exists and pins every report. The forgotten-publish gate (pre-push hook and CI's "Reports published" step) is inert until a manifest is written; once one exists, a PR that touches a report with no pin trips the gate, so the first publish has to cover them all. Pins need a git-backed publish tier (`[tool.mini] publish-repo`; the single-bucket default has no history to pin), and the `skip-publish-check` label has to exist on the repo.

## Library findings from the backport review (2026-09-03)

Found while reviewing the ported `src/mini` and inherited from sca2, so any fix belongs in both trees. Ordered by how much they matter.

- **A watchdog abort can be overwritten on Modal.** The stall handler (`abort_stalled` in `src/mini/_taskworker.py`) settles FAILED through `merge_if`, which on Modal is a read-modify-write with no lock. The progress emitter thread is still running, so a merge that read before the FAILED write and lands after it puts the state back to running, and then the process exits. The record eventually settles as "worker vanished" on reap, but the stall diagnosis is orphaned and the monitor agent takes the wrong branch. Fencing the emitter before the terminal write would close it.

- **The interactive Modal path rejects mini's kwargs.** `_build_modal_fn` (`src/mini/modal_apparatus.py`) pops only `startup_timeout`, while the memo path also drops `watchdog`, `watchdog_grace`, and `name`. So `.w(watchdog=600)` followed by `map` raises a TypeError from Modal. Not reachable from the CLI.

- **Non-string `[tool.mini] env` values wedge a local run.** The subprocess spawn (`src/mini/local_apparatus.py`) raises on an int after the record was already claimed RUNNING with no pid, so reap never settles it and only `cancel` clears it. Modal rejects the same config with a clean error; the local backend should validate up front too.

- **Plain `watch` omits the numerics-drift note** that `status` and `watch --json` both carry (`src/mini/__main__.py`).

- **`--watchdog-grace 0` aborts every task before its first emission.** The stamp in `src/mini/_taskworker.py` treats 0 as unset, but the value still reaches the watchdog.

- **Property bodies aren't walked by the memo fingerprint.** `_collect_class` (`src/mini/memo.py`) recurses into plain functions, staticmethods, and classmethods, so a deferred import inside a `property` or `cached_property` leaves the memo key unchanged. The one place the "deferred imports are traced" claim doesn't hold.

- **Unverified: the stall handler does Volume I/O before the hard exit.** If that I/O blocks on a wedged container, the exit never runs. Worth a bounded timeout or an exit-first ordering.

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

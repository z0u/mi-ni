# Todo

Scratchpad for deferred work that isn't worth a tracking issue yet. When something
here grows real, promote it to a GitHub issue and remove it from this list.

Right now this file is empty of scratch items — everything below is the
prioritized index into GitHub issues. Durable design rationale and recorded
decisions live in [`research/design.md`](./research/design.md); each open issue
also carries a grounding comment with current file:line refs, so it should be
readable cold without re-deriving code state.

## Backlog, grouped by what a single dev session should bundle

**Quick wins.** All shipped: #39 and #36 (PR #51), #19 (queued ≠ running,
PR #54), #47 (per-experiment backend memory for `--app`).

**Storage/control-plane design — read together, ship independently (maybe).**
These stem from the same list in `research/design.md`:

- #37 — implicit cross-experiment memo dedup. Bigger of the two: requires
  tag-scoping the whole control plane (`cancel`/`retry`/budget/`__run__`
  metadata) so a shared store doesn't let one experiment's teardown hit
  another's in-flight tasks. Read alongside #46 — a shared working volume
  (#37's optional sub-goal) reintroduces the same mutable-name hazard #46
  describes, at cross-experiment scope.
- #38 — publish-tier hardening (private-CAS/public-publish bucket split;
  citable versioned publish via a dataset repo). Independent of #37; only
  matters once the template is used for work that shouldn't be world-readable
  by default.
- #46 — fence mutable-name writes against stale workers. The `set_ref`/`publish`
  half is done (gen-fenced ambient store in the worker; `StaleWriteError`); the
  `get_data_dir()` half stays open only insofar as #37's shared volume would
  make it cross-experiment.

**Sequence after the above:**

- #15 — GC across the control plane, I/O-plane volume dirs, and the CAS. The
  local per-experiment control-plane + I/O-plane sweep shipped as `mini gc`
  (PR #49); Volumes confirmed to persist indefinitely (no Dict-style expiry;
  per-path `rm` exists), so the remaining legs are the Modal Volume sweep and
  CAS refcounting — the latter's shape still depends on how #37 (shared
  store?) and #38 (bucket split?) land.

**Orthogonal, no code overlap with the above:**

- #45 — docs rework. Touches `docs/`, `README.md`, `research/`, not `src/mini/`.
  Can run in parallel with anything.
- #57 — CLI DevX: passing a name to `retry`/`run` dies with a raw traceback
  (tick verbs take a file, read verbs a name). Tier 1 (friendly error + help
  text on the `path` positional) is a quick win in `src/mini/__main__.py`.

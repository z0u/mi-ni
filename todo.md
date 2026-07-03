# Todo

Scratchpad for deferred work that isn't worth a tracking issue yet. When something
here grows real, promote it to a GitHub issue and remove it from this list.

Right now this file is empty of scratch items — everything below is the
prioritized index into GitHub issues. Durable design rationale and recorded
decisions live in [`research/design.md`](./research/design.md); each open issue
also carries a grounding comment with current file:line refs, so it should be
readable cold without re-deriving code state.

## Backlog, grouped by what a single dev session should bundle

**Quick wins — small, independent, safe to pick up in any order or together.**
Each is a one- or two-file change with no interaction between them:

- #39 — wire `mini.store` through the interactive `app.map`/`arun` path
  (`store_context` is entered on the detached path but not `_wrap_for_local`/
  `_wrap_for_modal`). Real bug, not just a gap.
- #19 — surface "queued, never started" distinctly from "running" (the
  `state==RUNNING` + no `env` signal already exists, just isn't surfaced in
  `status`/`watch`). Diagnostic only.
- #36 — TLS workaround for the `modal` CLI in Claude Code Cloud (same fix
  already done for our Python code via `mini._tls`, needs the CLI equivalent).
  Worth prioritizing disproportionately: it's blocking Claude's own ability to
  diagnose Modal jobs, i.e. it pays back in every future session.
- #47 — remember (or default) which backend an experiment is running on.
  `--app` silently defaults to `local` on every subcommand, so `run --app modal`
  then `status` (no flag) reads the wrong store and prints a bare "no
  tasks found" with no hint. Confirmed footgun, no mitigation today.

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
- #46 — fence mutable-name writes (`set_ref`/`publish`/`get_data_dir()`)
  against stale workers.

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

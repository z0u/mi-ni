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

**Storage/control-plane design.** These stem from the same list in
`research/design.md`:

- #38 — publish-tier hardening (private-CAS/public-publish bucket split;
  citable versioned publish via a dataset repo). Only matters once the template
  is used for work that shouldn't be world-readable by default.
- Settled: #46 shipped (gen-fenced `set_ref`/`publish` + `StaleWriteError`,
  PR #56). #37 (implicit cross-experiment dedup + shared working volume) closed
  as not planned — the explicit ref path covers reuse; reopen only if
  identical-prep recompute becomes a real recurring cost.

**Sequence after the above:**

- #15 — GC across the control plane, I/O-plane volume dirs, and the CAS. The
  local per-experiment control-plane + I/O-plane sweep shipped as `mini gc`
  (PR #49); Volumes confirmed to persist indefinitely (no Dict-style expiry;
  per-path `rm` exists), so the remaining legs are the Modal Volume sweep and
  CAS refcounting — the latter's shape depends only on how #38 (bucket split?)
  lands, now that #37 is closed (no shared working store to anticipate).
  The `mini-hf-cache` Volume (#50) barely expands this: its `xet/` half is
  size-capped by `hf_xet`'s default, `hub/` grows only per distinct upstream
  model, and `modal volume delete mini-hf-cache` is always a safe reset (pure
  cache). Worth a mention in the sweep docs, not a new GC leg.

**Orthogonal, no code overlap with the above:**

- #45 — docs rework. Touches `docs/`, `README.md`, `research/`, not `src/mini/`.
  Can run in parallel with anything.
- #57 — CLI DevX: passing a name to `retry`/`run` dies with a raw traceback
  (tick verbs take a file, read verbs a name). Tier 1 (friendly error + help
  text on the `path` positional) is a quick win in `src/mini/__main__.py`.

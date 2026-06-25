# Todo

## Artifact store / sharing (#13, #22) — follow-ups

The content-addressed artifact store (`mini.store`) landed with `LocalStore` +
`HFStore`, project-scoped sharing, named refs, and `publish` (see
`.agents/skills/mi-ni/references/storage.md`). `HFStore` is selected by
`MINI_STORE_BUCKET`; the Modal worker gets it via a forwarded Secret, so
cross-experiment artifact sharing works on Modal **without** a shared Volume.
Deferred, in rough priority order:

- **Separate publish bucket / private durable store.** We use one public bucket
  for both durable CAS and published views (HF free tier = one bucket). `publish`
  is a separate verb, so pointing it at a second (public) bucket while the CAS
  goes private is a small change when wanted.

- **Versioned publish tier.** A bucket gives immutable *content* (via `cas/<sha>`
  views) but not immutable *names with history*. If we ever need to cite a frozen
  figure, the cheap option is an append-only ref log (`refs/<name>/<run-id>`) over
  the same bucket before reaching for a dataset repo. (Open question from
  `research/artifact-store.md`: bucket vs. dataset repo.)

- **Streaming → promote-to-hash.** For chunk-by-chunk writers (Zarr, activation
  dumps), stream to a working path hashing incrementally, then server-side
  copy-by-xet-hash each chunk to `cas/<sha>` and assemble the tree. The seam is
  there (`HFStore` already copies by xet hash for `publish`); not built yet.

- **Modal gotcha — stale serialized worker.** A long-lived detached app can serve
  a previously-serialized `_modal_task_entry` (so store-wiring edits don't take
  until its containers drain). Surfaced while testing: a fresh app name or
  stopping the app fixes it. General to the memo worker, not the store; worth a
  note in running.md and maybe a deploy-version bump on worker-code change.

- **Interactive `amap` store wiring.** The blocking notebook path
  (`app.map`/`arun`) enters `data_dir_context` but not `store_context`, so
  `mini.store.put`/`get` only work on the memoized path. Wire it through
  `_wrap_for_local`/`_wrap_for_modal` if notebooks want artifacts directly.

- **Project-scoped Modal Volume (#22).** With HF backing the store, artifacts no
  longer *need* a shared Volume. The #22 volume change is now only for sharing
  *non-artifact* volume bytes (raw working files via `get_data_dir()`, checkpoints)
  and for memo-level compute dedup — and it still drags the control-plane cascade
  with it: `cancel`/`retry`/budget teardown and the `__run__` metadata must become
  **per-experiment tag-scoped** (a shared store spans all experiments, so an
  un-scoped `cancel` would settle every experiment's tasks). Rewrite
  `test_budget.py::test_budget_is_scoped_per_experiment` to assert tag-scoping.
  Recommendation (see PR discussion): consider keeping Volumes isolated and
  treating the artifact store as the sharing surface, so this cascade stays
  optional.

- **Implicit memo-level dedup (#22).** Today cross-experiment reuse is *explicit*
  (a named ref handed off via the shared store). The implicit version — experiment
  B memo-*hits* A's identical prep with no ref and no recompute — needs the memo
  resolution to consult a project-scoped, fingerprint-keyed result store, with the
  per-experiment `MemoStore` demoted to a run's control plane. Depends on the
  tag-scoping above.

- **Garbage collection (#15).** A CAS only grows. Need refcounting from live memo
  results + a sweep. Backend-shaped: a bucket has server-side `rm`; a Modal `Dict`
  has only `clear()`, no per-key delete — confirm the reclamation story under the
  project-scoped namespace.

- **Smaller:** whether `Store` should be async to match `Volume`; the on-disk
  manifest format for trees if it ever needs to be read without Python; and a
  `mini publish` / artifact-aware CLI surface if reports want one.

## Unrelated nit spotted

- `utils.time.duration` accepts `min`/`h`/`d` but not `m`; the `mini` CLI help and
  a couple of docstrings advertise `--budget 30m`, which raises. Either add an `m`
  alias or fix the docs to `30min`.

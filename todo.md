# Todo

## Artifact store / sharing (#13, #22) — follow-ups

The content-addressed artifact store (`mini.store`) landed with `LocalStore`,
project-scoped sharing, named refs, and `publish` (see
`.agents/skills/mi-ni/references/storage.md`). Deferred, in rough priority order:

- **`HFStore` — web-reachable publish (#13 publish tier).** `publish` currently
  returns a `file://` URL from `LocalStore`. A Hugging Face bucket backend would
  return an `https://` resolve URL for the same handle (extension drives
  `Content-Type`), moving large report assets off Git LFS. Blocked on: adding
  `huggingface_hub` (not installed) + the bucket egress allow-list
  (`*.xethub.hf.co`, `*.cdn.hf.co`); the bucket-vs-dataset-repo choice is still
  open (see `research/artifact-store.md`). Keep `HFStore` backend-swappable so the
  publish tier and a future durable working store can collapse onto one bucket.

- **Project-scoped Modal Volume (#22).** The artifact CAS rides the experiment's
  Modal Volume, so cross-experiment sharing on Modal only works within one
  experiment's Volume today (locally it's already project-wide). Repointing the
  Volume from experiment-name to project-id is the #22 working-store change — but
  it drags the control-plane cascade with it: `cancel`/`retry`/budget teardown and
  the `__run__` metadata must become **per-experiment tag-scoped** (a shared store
  spans all experiments, so an un-scoped `cancel` would settle every experiment's
  tasks). Rewrite `test_budget.py::test_budget_is_scoped_per_experiment` to assert
  tag-scoping rather than per-store isolation. This is the large, risky half;
  landed separately on purpose.

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

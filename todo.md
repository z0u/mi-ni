# Todo

Scratchpad for deferred work that isn't worth a tracking issue yet. When something
here grows real, promote it to a GitHub issue and remove it from this list.

The storage / artifacts / publishing backlog has moved out of here:

- Durable design rationale and recorded decisions: [`research/design.md`](./research/design.md).
- Scheduled work: GitHub issues — #37 (implicit cross-experiment memo dedup),
  #38 (private-CAS / public-publish split + citable versioned publish),
  #39 (wire the store through interactive `app.map`/`arun`), #15 (GC), #19 (queued vs.
  running visibility).

- **Late writes from stale workers — mutable names only.** The record/result
  half of this is fixed: attempts carry a generation stamp, workers fence every
  record write on it, and results land in gen-qualified files — a stale worker
  (superseded relaunch, or surviving `cancel`) can no longer overwrite its
  successor's state or result, and the double-spawn race in `Ctx._classify` is
  closed by the conditional claim in `mark_running`. What remains: a stale
  worker can still last-writer-win a mutable _name_ — `set_ref`, `publish`, and
  shared volume paths (`get_data_dir()` filenames). CAS blobs are immune.
  Possible fixes if it bites: per-task scratch dirs, generation-stamped refs.
  Auto-cancelling stale workers at tick time is still wrong — mid-run the
  requested-set manifest is only a lower bound. The guard remains: `cancel` (and
  confirm dead) before editing. Note the fence is airtight locally (flock) but
  best-effort on Modal (`modal.Dict` has no compare-and-swap) — see the
  backend-consolidation discussion if this needs to be exact.

  z0u: `modal.Dict` does have a synchronization primitive:
  `put(...,skip_if_exists=True)`. "Returns `True` if the key-value pair was
  added and `False` if it wasn’t because the key already existed and
  `skip_if_exists` was set." See https://modal.com/blog/cache-dict-launch

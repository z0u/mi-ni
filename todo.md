# Todo

Scratchpad for deferred work that isn't worth a tracking issue yet. When something
here grows real, promote it to a GitHub issue and remove it from this list.

The storage / artifacts / publishing backlog has moved out of here:

- Durable design rationale and recorded decisions: [`research/design.md`](./research/design.md).
- Scheduled work: GitHub issues — #37 (implicit cross-experiment memo dedup),
  #38 (private-CAS / public-publish split + citable versioned publish),
  #39 (wire the store through interactive `app.map`/`arun`), #15 (GC), #19 (queued vs.
  running visibility).

- **Late-orphan writes to mutable names.** A superseded worker (fn edited under a
  live run, against the hotfix rules) runs to completion and can last-writer-win a
  mutable name *after* its replacement already wrote it: `set_ref`, `publish`, and
  shared volume paths (`get_data_dir()` filenames). Memo results (per-key dirs) and
  CAS blobs (content-addressed) are immune. Auto-cancelling orphans at tick time is
  wrong — mid-run the requested-set manifest is only a lower bound, so it would kill
  in-flight downstream tasks and settle them CANCELLED (terminal). Options if it
  bites: per-task scratch dirs on the volume, or a generation stamp on refs. For now
  the guard is the existing rule: `cancel` before editing.

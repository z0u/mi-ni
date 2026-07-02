# Todo

Scratchpad for deferred work that isn't worth a tracking issue yet. When something
here grows real, promote it to a GitHub issue and remove it from this list.

The storage / artifacts / publishing backlog has moved out of here:

- Durable design rationale and recorded decisions: [`research/design.md`](./research/design.md).
- Scheduled work: GitHub issues — #37 (implicit cross-experiment memo dedup),
  #38 (private-CAS / public-publish split + citable versioned publish),
  #39 (wire the store through interactive `app.map`/`arun`), #15 (GC), #19 (queued vs.
  running visibility).

- **Late writes from stale workers.** A worker launched under old code (fn edited
  under a live run, against the hotfix rules) runs to completion and can
  last-writer-win a mutable name *after* its replacement already wrote it:
  `set_ref`, `publish`, and shared volume paths (`get_data_dir()` filenames). CAS
  blobs (content-addressed) are immune. Under identity keys the hazard extends to
  the *record and result themselves*: the stale worker shares its successor's key,
  so if it survives `cancel` (ignored SIGTERM) it can overwrite `result.pkl` and
  merge DONE over the new attempt's RUNNING. A generation stamp on attempts (also
  wanted to close the double-spawn race between the `state` read and
  `mark_running` in `Ctx._classify`) would fence both; per-task scratch dirs
  cover the volume paths. Auto-cancelling stale workers at tick time is still
  wrong — mid-run the requested-set manifest is only a lower bound. For now the
  guard is the existing rule: `cancel` (and confirm dead) before editing.

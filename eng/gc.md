# Reclaiming storage: `mini gc`

*Part of the [engineering notes](./README.md).*

Every experiment leaves durable state on **three planes**, and nothing reclaimed it:
the **control plane** (per-experiment records — local JSON, or a named `modal.Dict`),
the **I/O plane** (per-experiment result dirs — local files, or a Modal Volume), and
the project-wide **CAS** (`cas/<sha>` artifact blobs). `mini gc` sweeps all three, in
two scopes, and judges collectibility against the store's own invariants — never age or
size (an old DONE hit is exactly what memoization is *for*).

**Per-experiment (`mini gc <name>`)** sweeps one experiment's memo state on whichever
backend it ran on: superseded records with their result dirs and staged calls, stale
attempt files from replaced generations, orphaned result dirs, and staged calls for
settled tasks. It runs the same plan logic over a `GcIO` adapter — `LocalGcIO` over the
filesystem, `ModalGcIO` over one recursive Volume `listdir` + per-path `remove_file`.
Two invariants carry the safety: a **current** record is never collectible (a DONE one
is a future memo hit; even a FAILED one is live state — deleting it would silently
convert a terminal failure into a relaunch on the next wake), and a **superseded** one
is collectible only once the requested-keys manifest is trustworthy — the last tick ran
the DAG to completion (`complete` in the run meta) and nothing is unsettled. Age- and
keep-last-N retention from the original issue were deliberately **not** built: the
current/superseded split is the sound signal, and age is a poor proxy for it.

**Project CAS (`mini gc --store`)** is a **mark-and-sweep**. *Mark* walks every
experiment's records — current *and* superseded, both backends — plus every ref; *sweep*
deletes blobs nothing reaches that are also older than a grace window. Three design
points:

- **A forward artifact index makes mark cheap.** Each attempt writes a tiny
  `result-<gen>.artifacts.json` sidecar next to its result, listing the blob shas the
  result references (found by `artifact_shas`, an object-graph walk pruned at
  code/module boundaries; a tree's own sha is a manifest, not a stored blob, so only its
  file children count). Mark reads that JSON instead of unpickling every result — no
  project imports, no arbitrary code, one small read per record however large the result.
  Unpickling stays the fallback for pre-sidecar records only.
- **`set_ref` is the pin.** Refs are mark roots, so an artifact handed off by a stable
  name survives even with no record referencing it — that's the documented way to keep a
  blob alive across record gc. The store sweep never second-guesses the memo layer:
  *every* record is a root, superseded included, so collecting a superseded record's
  blobs is `mini gc <name>`'s call to make first.
- **Fail closed, plus a grace window.** The mark phase aborts the whole sweep
  (`StoreGcError`, nothing deleted) on any RUNNING/PENDING task (its worker may have just
  seen `has(sha) == True` for bytes it is about to reference), any unreadable result
  (references unknown), or any unknown `.app` backend stamp. On top of that, an
  unreferenced blob younger than `--grace` (default `14d`, git's prune horizon) is kept —
  the margin against writers this checkout can't see: an unpushed colleague's records, or
  a `put` that skipped its upload because the blob already existed moments before the
  sweep judged it garbage. Both scopes are **dry-run by default**; `--apply` deletes.

**Backend specifics.** Modal `Dict` entries self-expire after ~7 idle days, so the Modal
*control* plane largely cleans itself — which means a Volume result dir can outlive its
record; that orphan dir is the normal end state, collectible by design, not a race.
Volumes themselves never expire (no inactivity policy), so the pressure is on the Volume
result dirs and the CAS — both of which `mini gc` now reaches. `HFStore.delete_blobs`
purges the local warm cache alongside the bucket delete, so a stale cached copy can't
make `has()` claim bytes the bucket no longer holds (a *different* machine's warm cache
is the one caveat the grace window covers rather than eliminates). The workspace-wide
`mini-hf-cache` Volume (#50) is pure cache — `modal volume delete mini-hf-cache` is
always a safe reset — so it's out of gc's scope entirely.

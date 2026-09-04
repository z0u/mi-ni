---
status: open
tags: [modal, storage]
opened: 2026-07-26
---
# A settled state can land on a successor's attempt, and the reader then trusts it

`merge_if` on Modal is read-check-write with a one-round-trip window (`ModalRecordStore.merge_if`), so a superseded worker can merge `state=DONE` onto a record its successor now owns. The record then reads `gen=B, state=DONE` while `result-B.pkl` doesn't exist yet, and `Ctx._classify` → `store.result` raises `FileNotFoundError` out of the map. Cosmetic for progress fields (overwritten a second later); real for the one terminal write per task. Long-standing — the previous three-round-trip version had a wider window. Not fixable with `modal.Dict` primitives: `put(skip_if_exists=)` is insert-if-absent, which arbitrates creating a key (`write_if` already uses it for the double-spawn race) but can't compare-and-swap a value. Building CAS from it means a lock — 4+ round-trips on the hottest write in the system, and a worker dying mid-lock wedges the record forever unless the value carries a lease, whose expiry can't be stolen safely without… CAS. Cheaper where it counts: don't trust `state=DONE` without the matching gen-qualified result. A missing one means "the current attempt hasn't finished", which is exactly true, and it covers the race however it arose. Care needed — a missing result can also be a swept blob or a Volume commit that didn't land, and reading that as "still running" suspends the DAG forever (`reap_dead` only settles records that say RUNNING), so it probably wants a record reset so the next tick relaunches. Touches `keep_stale`, `retry`, and the `settled` aggregates; worth its own change.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/settled-state-lands-on-successor.md`](https://github.com/z0u/sca2/blob/main/todo/eng/settled-state-lands-on-successor.md) there) with the code it describes. [A watchdog abort can be overwritten on Modal](./watchdog-abort-overwritten-on-modal.md) is the sibling race, exercised by a thread in the same process.

---
status: open
tags: [library, modal, storage]
opened: 2026-09-03
bundle: inherited-from-sca2
priority: high
---
# A watchdog abort can be overwritten on Modal

The stall handler (`abort_stalled` in `src/mini/_taskworker.py`) settles FAILED through `merge_if`, which on Modal is a read-modify-write with no lock (`ModalRecordStore.merge_if`). The progress emitter thread is still running when it does, so a merge that read the record before the FAILED write and lands after it puts the state back to running, and then the process exits. The record eventually settles as "worker vanished" on reap, but the stall diagnosis is orphaned and the monitor agent takes the wrong branch. Fencing the emitter before the terminal write would close it.

It is a sibling of the superseded-worker race in sca2's [`settled-state-lands-on-successor`](https://github.com/z0u/sca2/blob/main/todo/eng/settled-state-lands-on-successor.md): the same one-round-trip window, exercised by a thread in the same process rather than by an earlier attempt.

Found while reviewing the ported `src/mini` in the 2026-09-03 backport and inherited from sca2's tree, so the fix belongs there first and then back here.

# Todo

## Liveness: distinguish "queued, never started" from "running" (follow-up to #14)

The wall-clock budget (#14) is the backstop that catches a task which is queued
on Modal but never scheduled (it otherwise looks alive forever and wedges
`--watch`). A smaller, separate refinement remains: improve the liveness probe
(`Apparatus._is_task_alive`) to *surface* "queued" distinctly from "running", so
a wedged-in-queue task is visible before the budget fires. Tracked alongside #14.

## GC / retention for stale resources (#15)

Reclaim control-plane records, named Modal `Dict`s (`mini-cp-<name>`), and Volume
`_memo/<key>/` dirs. A `mini gc` verb with a retention policy (age / keep-last-N /
explicit selection), a dry-run/`--list` mode, and the safety invariants (never GC
in-flight tasks; never drop a DONE result a live experiment still memo-hits).
The budget's control-plane sidecar (`META_KEY`) and `enforce_budget` pattern from
#14 are a reasonable precedent for where a scheduled GC could hook in.

# TODO

Deferred design/work items. See [the proposal](./docs/proposals/automated-research.md)
and [notes/agentic-experiments.md](./notes/agentic-experiments.md) for context.

## Memoized orchestration

- **Settled vs. retryable failure (don't busy-loop, don't hide).** Today
  `_classify` treats `FAILED` like "never run" and relaunches it every `tick`, so
  a deterministically-failing task busy-loops (and wedges its `map` — see
  `allow_partial` below). Fix: make `FAILED` *terminal/settled*; `tick`
  auto-launches only un-run / retryable tasks. Re-running a `FAILED` task takes
  intent: bump `version=`, edit the fn (new key), or an explicit retry lever
  (the `mini retry` the CLI no longer has — see "memo CLI gaps" below).

  We deliberately do **not** classify transient (preemption / OOM / network
  timeout) vs. fatal in code — it's context-dependent and a wrong guess either
  masks a bug or wastes money. The *agent* is the classifier; our job is to (a)
  never busy-loop and (b) surface what it needs to decide:
  - default: a task that throws → `FAILED`, with the traceback on the I/O plane
    and `last_error` / `attempts` / `errored_at` on the record. Visible, not
    swallowed.
  - opt-in bounded auto-retry per step (`max_attempts=N`, exponential backoff via
    a persisted `next_attempt_at`) for steps the author *knows* are flaky.
    Bounded + backed-off so it can't hammer; on exhaustion → `FAILED`.
  - `mark_running` rewrites the record wholesale, so it must carry `attempts`
    forward (today it'd clobber the counter).
  Document (skill + README + test).

- **`allow_partial=` for `ctx.map`.** Today `map` raises `Pending` if *any* item
  isn't `DONE`. Once failures settle (see above), a `FAILED` item no longer
  relaunches but still blocks the map — we want a way to proceed on partial
  results. Subtlety to design around: downstream code often `zip`s configs with
  results, so we can't just drop missing items (it'd misalign). Options: return
  positional sentinels (`None`/a `Missing` marker) for un-`DONE` items, or return
  a config-keyed mapping instead of a positional list. Decide and document
  (skill + README + test).

- **Document the fix/prune/retry semantics** (skill + README + test):
  - remove an item from configs → not requested; unchanged items are memo hits.
  - change an item (fold the seed into args) → new key → only that item re-runs.
  - a `FAILED` item is *terminal*: it does not auto-relaunch. Retry on purpose
    (`version=`, edit the fn, or an explicit retry) — see settled-vs-retryable
    above. (Opt-in `max_attempts` may auto-retry *before* settling, bounded.)
  - force a re-run of a `DONE` item via `version=` or by editing the fn.

## Polling / monitoring

- **Memo CLI gap: `retry`.** The CLI is now memo-only and name-addressed
  (`run`/`ls`/`status`/`results`/`logs`/`cancel`); the old run/job model (model 2:
  `submit`/`Run`/`mini launch`) is gone. `retry` is the one verb still missing:
  today re-running `mini run <path>` relaunches un-run/`FAILED` tasks (it
  busy-loops on `FAILED` — see settled-vs-retryable). Once `FAILED` is terminal,
  add an explicit `mini retry <name> [<key>]` lever.

  - ~~`cancel`~~ **Done.** `Apparatus.cancel(store)` marks unsettled tasks
    `CANCELLED` and delegates the per-task stop to `_stop_task`: local SIGTERMs the
    worker's process group (pid recorded at `spawn_tasks`, `pgid == pid`), Modal
    cancels the `FunctionCall` by `fc_id`. CLI `mini cancel <name> [--app]`.

- **Keep `tick` (drive) distinct from polling (read).** `tick` re-runs `main` and
  *launches* missing/retryable work — it has side effects. A status/monitor check
  must use the read-only path (`state` / `records`), never re-`tick`, so "is it
  done yet?" can't accidentally relaunch work. Worth stating explicitly in the
  skill so an agent doesn't poll by re-ticking. The `--watch` driver
  (`mini.monitor.drive_and_watch`) honours this: it `tick`s only to advance to the
  next stage, then polls `store.records()` read-only between ticks.

- **Cheap polling for large sweeps.** Cache settled (`DONE`/`FAILED`/`CANCELLED`)
  records client-side — they're immutable — and poll only the unsettled subset.
  Keep control-plane records small (latest scalars; history to the I/O plane).
  With per-job keys, read only active jobs; consider a last-writer-wins run-summary
  key for cheap top-level polling. Relevant once the control plane is a `modal.Dict`.

- **Interactive monitor.** *Partly done* for the memo path: `mini run <exp>
  --watch` (`mini.monitor.drive_and_watch`) drives the DAG to completion with a
  live Rich bar per task, looping on the durable read path (`MemoStore.records()`)
  — no `LocalQueue`. Ctrl-C stops only the watch; detached workers live on, so
  re-running resumes. Still to do:
  - A *read-only* `mini watch <exp>` (name-addressed, doesn't `tick`/launch) that
    renders the live bar for a run it didn't launch — i.e. `status` with a Rich
    refresh loop, for watching a detached/Modal run from another process.
  - **A dead worker wedges the watch.** A worker killed (or hard-crashed) without
    writing `FAILED` leaves a stale `RUNNING` record, and the drain loop waits on
    it forever. No heartbeat-timeout yet — and heartbeats only tick on
    `emit_progress`/state-changes, so a long no-emit step (e.g. `prepare_data`)
    can't be told apart from a stall by age alone. Tie this to the liveness
    cross-check under "Modal backend" (`FunctionCall.get(timeout=0)` there; a
    pid/heartbeat check locally).

## Compute decoupling (apparatus)

**Done (local-only slice).** Compute is injected at execution, not stored on the
experiment:
- `Experiment` no longer has an `apparatus` field (nor `make_apparatus`,
  `data_dir`, `before_hooks`); `tick(exp, apparatus)` takes it explicitly. CLI
  `--app`/`--workers` build it via `_build_apparatus`.
- Per-step override plumbed: `ctx.run(fn, *args, on=apparatus)` /
  `ctx.map(fn, items, on=apparatus)`, defaulting to the tick's apparatus; per-step
  `before_each` hooks come from whichever apparatus runs the step. Covered by
  `test_per_step_apparatus_uses_its_hooks`.
- Decided **against a generic `Requirements` type** (over-engineered: it would
  need a translation layer to each backend's native options — Modal's
  `function()` alone has gpu/cpu/memory/timeout/retries/region/…). The apparatus
  owns its native knobs; selection happens at the edge (CLI flag / notebook code).

**Still to do:**
- ~~**Wire compute through the memoized path.**~~ **Done.** `MemoStore` no longer
  spawns: it `stage`s the call durably and the apparatus spawns via the
  `Apparatus.spawn_task(store, key, call)` seam (`LocalApparatus` → subprocess;
  `ModalApparatus` → detached Modal spawn). `on=` now routes *compute*, not just
  hooks. Records went behind a `RecordStore` (local JSON / `modal.Dict`).
- **Role labels for file-based experiments.** `on=apparatus` only works from a
  notebook (which holds apparatus handles); a file experiment loaded by the CLI
  has none. Plan: abstract *role* labels (`ctx.map(..., role='gpu')`) that the
  driver/CLI maps to concrete apparatuses, so `main` stays backend-agnostic.
- **Capture compute-environment metadata in the run/task records** (what it
  actually ran on) — separate from backend selection above.

## CLI / ergonomics

- **Expose a real entrypoint.** *Partly done*: `bin/mini` wraps `uv run python -m
  mini …` so it works from any cwd (matches the `bin/` convention). Follow-up: a
  proper `[project.scripts]` console-script (`mini = mini.__main__:main`) so the
  wrapper can drop the `python -m` indirection — and decide whether the data root
  should follow the project (today `.mini` is relative to cwd, so `bin/mini` from
  elsewhere writes a stray `.mini/` there).

## Cross-experiment composition

- **Consume experiment A's output from experiment B.** Within one experiment the
  memoized DAG already handles `f → g`. For genuinely separate experiments, the
  clean path is value-level: import A's experiment and `tick` it (instant memo
  hit if done) to get its return value. Sugar like `get_data_dir(other)` for
  reaching another experiment's volume is the open question — lean toward
  explicit shared locations over reaching into another experiment's private dir.

## Modal backend

**Done (memoized path).** `mini run experiment.py --app modal` runs the memoized
orchestration detached on Modal, verified live on Modal 1.3.3:
- Control plane = named `modal.Dict` (`ModalRecordStore`, client-readable with no
  remote function). I/O plane = the Volume; the remote worker (`_modal_task_entry`,
  reusing the backend-agnostic `execute_task`) commits results before settling
  state. `ModalApparatus.spawn_task` spawns one detached call per task under
  `app.run(detach=True)` and persists the `FunctionCall` id.

**Monitoring validated live (2026-06-18).** End-to-end on Modal from this env:
`run --app modal` (detached launch + `fc_id`), `status`/`results --app modal`
(client reads of the Dict + Volume), `run --app modal --watch` (drove the DAG to
completion with live bars off the Dict), and `cancel --app modal` (cancelled both
`FunctionCall`s by `fc_id` → app went to `stopping...`).

**Still to do:**
- ~~**Read-only Modal commands shouldn't build the image.**~~ **Done.**
  `make_image` is now lazy (`ModalApparatus._ensure_image`, built once on first
  spawn/map), not eager in `__init__`. `status`/`results`/`cancel --app modal`
  touch only the `modal.Dict`/Volume, so they no longer run `uv` freeze or print
  "Creating Modal image…" (verified live: 0 occurrences on `status`).
- ~~**Client-side `gather` egress.**~~ **Worked live now.** `ModalMemoStore.result`
  reads result blocks from the Volume's storage CDN; this previously 403'd from
  the locked-down env, but `results --app modal` and `--watch`'s final gather both
  succeeded here (egress rules since configured — see the README network-egress
  section). Keep the remote gRPC read-back function in mind as a fallback if a
  more restricted env 403s again.
- ~~**`spawn_map` batching.**~~ **Done** (`Apparatus.spawn_tasks`). A `ctx.map`'s
  missing tasks launch in one detached `app.run`, and the memo worker drops the
  `max_containers=1` cap so the sweep parallelises (verified live: 3 tasks ran
  concurrently). We use one `spawn()` per task (not `spawn_map`) inside that
  single context: same batching win, but each task gets a `FunctionCall` id
  recorded in its memo record at launch — `spawn_map` returns `None` on 1.3.x,
  which would leave a failed launch undiagnosable. Follow-up: a programmatic
  liveness cross-check (`FunctionCall.from_id(fc_id).get(timeout=0)`) in
  `mini status`, distinguishing "queued but never started" from "running".
- ~~**Restricted-env TLS.**~~ **Done** (`mini/_tls.py`). Modal's gRPC uses
  `certifi`, which omits a corporate/sandbox proxy CA that lives in the system
  bundle → `CERTIFICATE_VERIFY_FAILED`. `ensure_grpc_trusts_system_ca()` (called
  from `ModalApparatus.__init__`) points `certifi.where()` at a certifi+system
  combined bundle, additively. Verified live on a clean certifi.

## Housekeeping (from the proposal's "Deferred / open")

- Auto-teardown on a wall-clock budget so a forgotten detached run can't burn
  money indefinitely.
- Garbage-collect old run records / named Dicts / Volume run dirs.

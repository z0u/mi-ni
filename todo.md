# TODO

Deferred design/work items. See [notes/agentic-experiments.md](./notes/agentic-experiments.md) for context.

## Memoized orchestration

- **Settled vs. retryable failure (don't busy-loop, don't hide).** *Core done.*
  `_classify` now launches only un-run tasks (`state is None`); `FAILED`/
  `CANCELLED` are terminal, so a deterministic failure no longer busy-loops (it
  does still wedge its `map` — see `allow_partial` below). Re-running takes intent:
  `mini retry` (resets FAILED/CANCELLED → `orchestration.retry` → `MemoStore.reset`),
  bump `version=`, or edit the fn. The traceback is on the I/O plane; the record
  carries `error` (last line).

  Still to do — the **opt-in bounded auto-retry** for steps the author *knows* are
  flaky, since we deliberately don't classify transient vs. fatal in code (the
  agent is the classifier; we just avoid busy-looping and surface the error):
  - `max_attempts=N` per step with exponential backoff via a persisted
    `next_attempt_at`; on exhaustion → `FAILED`. Bounded + backed-off so it can't
    hammer.
  - track `attempts` / `errored_at` on the record; `mark_running` rewrites the
    record wholesale, so it must carry `attempts` forward (today it'd clobber it).
  - Document (skill + README + test).

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

- **Memo CLI verbs.** The CLI is memo-only and name-addressed
  (`run`/`retry`/`ls`/`status`/`results`/`logs`/`cancel`); the old run/job model
  (model 2: `submit`/`Run`/`mini launch`) is gone.

  - ~~`retry`~~ **Done.** `mini retry <path> [--key K] [--watch]` resets
    FAILED/CANCELLED tasks (`orchestration.retry`) then advances the DAG; FAILED is
    terminal so a plain `run` won't relaunch. `run` prints a retry hint when a task
    settles FAILED.
  - ~~`cancel`~~ **Done.** `Apparatus.cancel(store)` marks unsettled tasks
    `CANCELLED` and delegates the per-task stop to `_stop_task`: local SIGTERMs the
    worker's process group (pid recorded at `spawn_tasks`, `pgid == pid`), Modal
    cancels the `FunctionCall` by `fc_id`. CLI `mini cancel <name> [--app]`.

- **Keep `tick` (drive) distinct from polling (read).** `tick` re-runs `main` and
  *launches* missing/retryable work — it has side effects. A status/monitor check
  must not re-`tick`, so "is it done yet?" can't accidentally relaunch work. Worth
  stating explicitly in the skill so an agent doesn't poll by re-ticking. The
  `--watch` driver (`mini.monitor.drive_and_watch`) honours this: it `tick`s only
  to advance to the next stage, then polls `store.records()` between ticks. The one
  sanctioned write on the poll path is `reap_dead` (settles a vanished worker to
  terminal `FAILED`); it never *launches*, so the no-relaunch invariant holds.

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
  - ~~**A dead worker wedges the watch.**~~ **Done.** `Apparatus.reap_dead(store)`
    cross-checks each `RUNNING` task against the *real* worker (`_is_task_alive`:
    local probes `/proc/<pid>/stat`, counting a zombie as dead so a SIGKILLed child
    of the watcher is caught; Modal probes `FunctionCall.get(timeout=0)`) and
    settles orphans `FAILED` ("worker vanished…"). It re-reads state before writing
    (a worker settles *then* exits, so gone+RUNNING ⇒ died mid-run) and never
    relaunches, so it's safe on the poll path. Wired into the `--watch` drain loop
    (`drive_and_watch`) and `mini status`, so a killed worker surfaces as `FAILED`
    (→ `mini retry`) instead of hanging. We sidestep heartbeat-age entirely (it
    can't tell a long no-emit step from a stall), probing the worker directly.
    Modal is deliberately conservative — only *definitive gone* signals
    (`OutputExpiredError`/`NotFoundError`) reap; transient/infra read errors are
    treated as alive, since a false "dead" would mark a live GPU task FAILED and a
    retry would double-spawn it.

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
  spawns: it stages each call durably (`write_call`) and the apparatus launches a
  whole batch via the `Apparatus.spawn_tasks(store, batch)` seam (`LocalApparatus`
  → subprocess per task; `ModalApparatus` → detached Modal spawn). `on=` now routes
  *compute*, not just hooks. Records went behind a `RecordStore` (local JSON /
  `modal.Dict`).
- ~~**Role labels for file-based experiments.**~~ **Done.** `ctx.run/map(...,
  role='gpu')` names a label; `Experiment(roles=...)` binds it. The common form is a
  table `{label: .w()-kwargs}` applied to whatever `--app` built (`resolve_roles`);
  a callable `(base) -> {label: apparatus}` is the escape hatch for per-role
  `before_each`/image. Backend-native knobs stay typed in code (not CLI strings),
  and the same table runs locally — `Apparatus.w` defaults to a no-op so local
  ignores Modal's `gpu=`. `role=`/`on=` are mutually exclusive; an undefined role
  raises. All roles share one volume by construction (`.w()`/`clone` keep `_volume`),
  so heterogeneous hardware still sees the same storage. Verified live on Modal
  (`docs/role-demo`: probe→cpu, gpu→L4).
  - *Deferred (separate primitive):* **run presets / parameterized `main`** — bundle
    {config overrides + role tier} under one name (`smoke` vs `full`) so the same DAG
    runs at different sizes. This is *not* roles: batch size / steps are
    hyperparameters (config, already swept + memoized), and since the memo key
    excludes hardware, a size bump must change the config, not just the GPU. `main`
    currently takes only `ctx`, so passing a preset in is the missing piece.
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
  state. `ModalApparatus.spawn_tasks` opens one detached `app.run(detach=True)` for
  the whole batch and `spawn()`s each task within it, persisting each task's
  `FunctionCall` id (see "spawn_map batching" below).

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
  which would leave a failed launch undiagnosable. The programmatic liveness
  cross-check (`FunctionCall.from_id(fc_id).get(timeout=0)`) now lands in
  `ModalApparatus._is_task_alive`, used by `reap_dead` from `mini status`/`--watch`
  (see "A dead worker wedges the watch" above). It does not yet distinguish
  "queued but never started" from "running" — both read as alive.
- ~~**Restricted-env TLS.**~~ **Done** (`mini/_tls.py`). Modal's gRPC uses
  `certifi`, which omits a corporate/sandbox proxy CA that lives in the system
  bundle → `CERTIFICATE_VERIFY_FAILED`. `ensure_grpc_trusts_system_ca()` (called
  from `ModalApparatus.__init__`) points `certifi.where()` at a certifi+system
  combined bundle, additively. Verified live on a clean certifi.

## Housekeeping (from the proposal's "Deferred / open")

- Auto-teardown on a wall-clock budget so a forgotten detached run can't burn
  money indefinitely.
- Garbage-collect old run records / named Dicts / Volume run dirs.

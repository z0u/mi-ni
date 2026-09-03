# Running & monitoring an experiment

An experiment runs detached and durable: work is launched as detached workers and its state, results, and errors are written to a per-experiment store. You drive and poll it from short-lived CLI processes (`bin/mini`), addressed by experiment name. Run `bin/mini --help` for the verb list.

The store lives at the project root (`.mini/`, found by walking up for `pyproject.toml`/`.git`), so the verbs work from any working directory: a poll from a subdir finds the same run. `bin/mini` wraps the `mini` console-script and pins the project, so it also runs from anywhere without an activated venv.

## The one invariant: tick vs. read

`run`, `retry`, and `cancel` _tick_ the DAG. They re-run `main` and launch (or stop) work, so they have side effects and cost money. `ls`, `status`, `watch`, `results`, `logs`, and `explain` only read the durable store, and are safe to call any time.

So poll with `status`, never by re-running. `--app modal` inspects a run on the Modal control plane. Don't pass `--watch` in a capped or agent session, since `run --watch` blocks to completion; one plain `run` launches the next stage and returns at once. To follow a run with a live bar *without* driving it, use the read-only `mini watch <name>`, which renders a run another process launched (a detached or Modal run) and never `tick`s.

The backend sticks. `run --app modal` stamps the launch backend into `.mini/<name>/.app`, and every later verb without `--app` follows it — launch with the flag once, then plain `status`/`watch`/`results`/`retry` hit the right control plane (to move a run's home, pass `--app` again; the marker updates). The marker is per-checkout (`.mini/` is untracked), so a fresh clone — CI, a scheduled monitor's new environment — has no memory: set `MINI_APP=modal` in the environment, or commit `app = "modal"` under `[tool.mini]` in `pyproject.toml`, to make the default travel (precedence: flag > marker > env > pyproject > local). Either way, a read that lands on an empty backend peeks at the other one and names the flag: `no tasks found for 'pipeline' on local — found 3 task(s) on modal — try: --app modal`.

## The wake-loop

A capped session can't babysit a long run, so work in wakes, where each verb is a cheap, stateless call against durable state:

1. Launch or advance: `bin/mini run <exp>` (returns immediately).
2. Later, poll: `bin/mini status <exp>` (read-only).
3. On failure: `bin/mini logs <exp> <key>`, fix, `bin/mini retry <exp>`.
4. When done: `bin/mini results <exp>`, or open `report.py`.
5. To ship the report: once it renders the results, `./go publish <report>`. Completion auto-publishes the *results* to the store, but the *report* bundle is a separate, deliberate step. The site build silently skips an unpublished report (a warning in CI logs, not an error), so it won't appear until you publish it.

Re-running is cheap: completed steps are memo hits, so a `run` only advances the un-run pieces.

To wait for a stage to settle in an agent session that wants one wake-up rather than a sleep loop, use `bin/mini watch <exp> --timeout 10m --json`. It polls read-only, reaps vanished workers, and exits the moment there's something to act on. The exit code names the branch, so no output parsing is needed to decide: `0` means the current tasks all settled DONE, so `run` again to advance the DAG (the tick prints `✓ complete` when nothing is left); `1` means settled with FAILED or CANCELLED, so `logs` and `retry`; `124` means `--timeout` elapsed with work still in flight, so re-watch or report progress.

`3` means attention: a task settled terminally *mid-stage* (a watchdog fired, a vanished worker was reaped) or a RUNNING task's liveness went stale (stale heartbeat, frozen step). Act now instead of waiting out the siblings. Pre-existing terminal tasks don't trigger this, only events that happen during the watch.

With `--json` the live bars are replaced by one compact summary object (`outcome`, `reason`, `counts`, and an `attention` list). So the whole wait is one command: never hand-roll a `while`/`sleep` polling loop, and never `grep` the human lines. For a one-shot snapshot, `bin/mini status <exp> --json --brief` gives the aggregate `state`/`settled`, counts by state, and *only* the tasks needing attention — token-cheap on a big sweep. Full `status --json` (per-task `state` / `queued` / `heartbeat_age_s` / `stale_heartbeat` / `progress_age_s` / `stale_progress` / `steps_per_min`) is for digging into one task. Bound any wait with `--budget` so an unattended stall settles itself.

### What counts as needing attention

Failures and liveness are the obvious half: terminal-not-DONE, a stale heartbeat, a frozen step, a task queued past the staleness window. The other half is deviation from expectation, since a run can misbehave while every liveness check reads green, so `status` does the comparing and puts a `cause` on each attention entry:

- *N steps/min — under a third of the sibling median (M)* — the cells of a fan-out are each other's yardstick, so a container placed badly (or throttled) shows up against the median `steps_per_min` of its siblings. Needs three reporting siblings before there's a distribution to speak of, and it assumes the cells do comparable work: a `map` whose args vary the amount of work per cell will flag its big cells, so read the flag against the args.
- *projected Nm to finish, past its Mm timeout* — remaining steps ÷ current rate, against the role's `timeout=`. A timeout sized as a multiple of the expected duration turns into a kill switch when throughput drops, and it fires near the end, after the work is paid for.
- *loss is not finite* / *accuracy falling for several windows* — from the metrics a task reports via `emit_metrics`. The worker compares window *means* (a per-step loss is noisy enough that two boundary samples are a coin flip) and counts how long each metric has run against its goal, so a trend is a number to check rather than a curve to eyeball: `metrics_delta`, `metrics_wrong_way` and `metric_goals` on the record.

That last one only works if the task emits its numbers as numbers: put them through `emit_metrics(loss=…)`, not into the `emit_progress` message string.

Which way is wrong is yours to say: `expect_metrics(loss="down", accuracy="up")`, once before the loop, and both directions read as the same alarm, since a sliding accuracy is as much of a problem as a climbing loss. Nothing downstream matches on metric names, because a name can't tell you whether a number is meant to climb (`loss_scale` climbs by design; a domain-specific score gives away nothing). The worker guesses only for a handful of unambiguous names — `loss`, `val_loss`, `nll`, `perplexity` and friends — and anything else is measured but never flagged until you declare it. The count needs one window more than its threshold before it can fire, since the first window only sets the baseline, so give a young run about four minutes before reading its trends.

Watching a big sweep is cheap too: the watch loops cache settled (`DONE`/`FAILED`/`CANCELLED`) records — they're immutable — and re-read only the tasks still in flight, so a mostly-done sweep stops paying to poll its settled tail (on Modal each record read is a `Dict` round-trip). Each task also records what it ran on (host/OS/Python, CPU/RAM, the GPU + count when attached, the numerics env it ran under — `XLA_FLAGS`, which decides whether a GPU reduction is deterministic — and on Modal the container id / region / cloud (never any token); `status` shows `on <GPU>` for remote tasks, and the full snapshot is on the record under `env`, with `started_at`/`finished_at` for a real execution duration.

## Where the containers land (`--region`)

Left unspecified, Modal places containers wherever it has capacity — which for one sweep can mean several continents. What that costs depends on *which* shared thing the task talks to, and the three behave differently:

- The control-plane `modal.Dict` is a synchronous RPC per call, so distance is paid as latency, once per call. This is the one we've measured: training cells placed outside `us-east` ran 15–30× slow, in order of distance, because a per-step progress emission blocked on it. Emission now runs off the task's thread, so that cliff is gone even unpinned.
- Volume writes are buffered and committed in bulk, and the Volume is eventually-consistent anyway, so distance costs one bulk transfer per commit rather than a round-trip per write. We have no measurement here, and the shape of the cost argues it's minor.
- Cold Volume *reads* fetch on open, so a role that reads a lot of data it hasn't cached is the most plausible remaining case for pinning — also unmeasured.

Pin it with `--region us-east` (or `[tool.mini] region` for a project-wide default, or `region=` on a role, which wins over both). The trade-off goes the other way too: pinning costs a per-region premium and narrows the capacity pool, so a pinned sweep can sit queued where an unpinned one would have started.

So the default is not to pin; reach for it when you have a reason. If you suspect placement is the problem, `env.region` is on every task record (`status --json`), and a cell far behind its siblings' throughput is flagged for you — measure before pinning, since the per-call chatter that made placement bite last time has moved off the task's thread.

## Provenance & cost

A run stamps lineage into its meta on every wake, enough to reproduce or forensically trace it: the git state (sha, branch, tags, sanitized remote, and the working-tree diff when the tree is dirty), who or what drove it (the AI agent(s) plus a non-PII operator handle — the repo owner from the remote, since the git `user.name` is a bot in agent and CI contexts and a real name is PII), the spawning environment, and the timeline. `bin/mini lineage <exp>` prints the summary (`--diff` dumps the recorded diff), including a rollup of what the tasks ran on. Upstream experiments are captured automatically: a step that `get_ref`s another experiment's ref records that producer, and the driver snapshots each producer's provenance into `lineage.upstreams` (shown as `⇐ <producer> … via <ref>`). Declare `Experiment(deps=[...])` only to force an upstream the run doesn't read via a ref (e.g. served from a memo hit, or handed over via the volume).

Every attempt also records the numerics packages it ran under (`jax`, `jaxlib`, `numpy`). A library upgrade moves a number without moving the memo key, so a finished record can keep serving a result the current environment wouldn't reproduce. A tick warns once when it serves such hits, and `status` says the same from the records alone: `⚠ N result(s) computed under different numerics: jax 0.10.1 → 0.11.0`. Nothing re-runs on its own — the levers are `version=` on the affected tasks, or saying so where the numbers are published (`eng/determinism.md`).

On Modal, each run's app-instance ids are recorded for cost attribution. `bin/mini cost <exp>` reconciles the run's spend from the billing API with a per-resource breakdown (CPU / Memory / each GPU type). It's a post-run query: Modal bills at daily resolution and lags the run, so a just-finished run reports nothing yet — check back later.

## Reading a task that isn't progressing

Three states look alike from the outside and want different responses.

### Queued is not running

A record reads RUNNING from launch, but the worker writes `env` as its first action — so until `env` appears, the task is *launched but not started*. `status` shows it as `◌ queued` with its time in queue (`⧖`) instead of a heartbeat, and `watch` tags its bar `— queued`. Locally this is a momentary blip; on Modal a capacity-starved task can sit queued indefinitely, and only the wall-clock budget (below) will reap it. A task stuck on `queued` with an old `⧖` is a scheduling problem (capacity, container boot), not slow code.

### Dead is not slow

The inverse failure: a RUNNING task whose progress is frozen and whose heartbeat (♥) has gone stale for minutes while its siblings beat every few seconds. That worker may be dead, most often killed by the role's per-task `timeout`. Any `status`/`watch` poll reaps a call Modal reports as settled (timeout-killed, terminated, expired) to FAILED; a heartbeat stale past ~5 minutes that *hasn't* been reaped gets an advisory badge (`⚠ stale — worker may be dead`, `stale_heartbeat` in `--json`), meaning either a worker in a long non-emitting stretch (a heavy import, one big step) or a liveness blind spot. A task inside a declared step-free span (`mini.blocking_phase`, which artifact transfers use for themselves) is exempt for as long as that span's budget lasts, so a step whose whole body is `put`/`get` calls — a publish step fanning in — stays quiet rather than badging: heartbeats ride on progress emissions, and it makes none. Size the timeout for the *largest* cell of a sweep, not the typical one. Note that `mini` does not set a timeout for you: a role without `timeout=` gets Modal's default of 5 minutes, and its default CPU slice (0.125 cores), where heavy imports (jax) can alone take minutes. Any role doing real work should set `timeout=` (and `cpu=` when CPU-bound). Recover deliberately: `cancel --key <key>` to stop just that worker and reap its record (plain `cancel` stops the whole run), raise the role's `timeout` (execution config — DONE cells stay memo hits), then `retry --key <key>`.

### Wedged is not dead either

A worker can also *wedge*: a hung device call or deadlocked thread that holds its GPU allocations while making no progress (seen in the wild: container alive, GPU at 0.3 % utilization for 45 minutes). The record tells the two signals apart: `heartbeat_at` advances on *any* emission, `progress_at` only when the step advances, so `progress_age_s` climbing while the heartbeat stays fresh is the wedge signature (`⚠ no step progress` in `status`, `stale_progress` in `--json`; `steps_per_min` gives the trailing throughput for "slowed down" vs "stopped"). The fix is worker-side: give the role a `watchdog=` (seconds without step progress) and a wedged worker settles itself FAILED with an all-thread stack dump (`WatchdogStall`) and exits — a fast, retryable failure instead of a silent burn until `timeout`. Until the first progress emission, `watchdog_grace=` applies instead (default: same as `watchdog`), so a long one-off setup phase doesn't force the watchdog loose — keep `watchdog` sized to the step cadence and let the grace cover the prep. Step-free spans mid-run declare themselves with `mini.blocking_phase` (artifact transfers already do), so a checkpoint upload after the last step isn't read as a wedge. Treat a `WatchdogStall` like a flaky infra failure unless it recurs on the same cell: `retry --key <key>`.

## Recovery

The fix loop — why `FAILED` and `CANCELLED` are terminal, the fix/prune/retry table, bounded hotfixes with `--keep-stale-done`, superseded records, and partial `map` failures — is in [recovery.md](./recovery.md). What follows here is its operational side: what an edit costs while workers are still live, and one Modal-specific way a fix can look like it was ignored.

### Hotfix safety (avoid double-spending)

Editing a task fn makes every cell of it *stale*, so it re-runs (in place — keys are identity, so nothing is orphaned). But a re-run does not kill workers already detached under the old code; they keep burning, which is real money on Modal. And editing a shared helper invalidates every task that calls it. Three rules keep the blast radius bounded:

1. Only hotfix terminal (FAILED/CANCELLED) tasks. Their worker is already dead, and a stale terminal task relaunches on the next `run` (no `retry` needed; the fix is what it was waiting for). For a transient failure, don't edit: `retry --key <key>` re-runs just that task, so the blast radius is one task. By default an edit also re-runs the fn's DONE cells — for a `map`, the whole fan-out. If the fix demonstrably doesn't change what the finished cells computed, run with `--keep-stale-done`: DONE results are served as-is (badged `(stale code — kept)` in `status`) and only the unfinished cells re-run.
2. If anything is in-flight, `cancel` first, then fix. Never edit under a live worker, or you pay for compute nobody wants. (`cancel` is store-scoped, so it also stops stale old-code workers, which keep showing as `RUNNING` in `status`.) The *records* are safe either way: every attempt runs under a generation stamp, so a stale worker that survives `cancel` can't overwrite its replacement's state or result. It can still race mutable *names* it writes, though (`set_ref`, `publish`, shared volume paths).
3. Only ever edit the single failing task fn. Never a shared helper, `main`, or the DAG shape, since those re-run an unbounded set of tasks. That is an escalation rather than a hotfix.

`cancel` is also the cost-control lever: stop in-flight work you no longer want.

### Modal: stale serialized worker

A long-lived detached Modal app can keep serving a *previously serialized* worker entry, so an edit to worker/store wiring won't take effect until its containers drain — the run looks like it's ignoring your fix. If a worker-side change isn't being picked up on Modal, stop the app (or launch under a fresh app name) to force a re-serialize. (This is a property of the detached app, not the artifact store.)

## Wall-clock budget (auto-teardown)

A detached run outlives the process that launched it, so a forgotten or wedged run can burn money (Modal) or hold local resources indefinitely. Bound the whole sweep with a wall-clock budget:

```
bin/mini run <exp> --budget 2h          # the run may not outlive 2 hours
bin/mini run <exp> --app modal --budget 30m
```

`--budget` stamps a `deadline_at` into the run's control plane at launch (a sidecar on the same store — local JSON / Modal `Dict` — so no new infra). There's no supervising process to fire a timer, so enforcement is opportunistic: any process that already touches the store — `status`, `watch`, the `--watch` driver — cancels in-flight tasks (→ `CANCELLED`) once the deadline passes, via the same `cancel` path. A driver also refuses to launch a *new* stage past the deadline. So a budgeted run that goes unattended settles cleanly the next time anything polls it; `status` shows `budget 2h, 12m left` (or `expired`).

The budget is run-level, complementing the per-task `--timeout` (Modal's function timeout, which bounds one task). Passing `--budget` again re-arms the deadline relative to now (so you can `retry` past an expired budget); a plain re-run to advance a multi-step DAG inherits the existing deadline. `cancel` is the manual, immediate version; the budget is the unattended backstop.

Size the first run's budget for the image build. In a fresh Modal environment the first launch spends minutes building the container image while the task sits `queued`, and the clock is running: a `--budget 10m` has expired before any work starts, and the next poll settles the run CANCELLED. The image is cached afterwards, so `retry --budget …` goes straight through — confusing rather than costly, but only if you know to expect it.

## Escalation contract

Attempt only a local, obvious fix on a terminal task (typo, bad path, wrong hyperparameter) within rules 1–3. Otherwise stop and report up rather than guessing past the mandate. Escalate when the fix isn't local and obvious, when the same task fails again after a fix, when the failure is in experiment design or `mini` internals, or when cost looks wrong (runaway relaunches). Report:

```
{ experiment, state summary, failing key(s), last error line,
  traceback excerpt, what I tried, recommended next step }
```

## Delegating & scheduling a long run

To launch and babysit a run without spending the main session's (expensive) context, delegate to the `experiment-monitor` subagent (Haiku): "poll status of `<exp>`, advance if asked, apply a bounded hotfix if a task failed obviously". It does one pass and reports. The Haiku monitor can't spawn other agents, so its escalation flows back to you: on an escalation report, spawn the `experiment-doctor` subagent (Opus). Bring a redesign to the human rather than reshaping the experiment yourself.

For a run too long to watch in one session, set up a scheduled routine (`CronCreate`, or the Claude_Code_Remote `create_trigger`/`send_later` tools — whichever this session offers) at a cadence the user picks; don't assume one. Each wake, the routine spawns the monitor (and, on escalation, the doctor, then notifies). It self-removes when the run settles: when `status` shows a terminal aggregate state, find the routine's id (`CronList` / `list_triggers`, match by name) and delete it (`CronDelete` / `delete_trigger`). A recurring cron costs money, so confirm with the user before creating it.

# Running & monitoring an experiment

An experiment runs **detached and durable**: work is launched as detached
workers and its state, results, and errors are written to a per-experiment
store. You drive and poll it from short-lived CLI processes (`bin/mini`),
addressed by experiment **name**. Run `bin/mini --help` for the verb list.

The store lives at the **project root** (`.mini/`, found by walking up for
`pyproject.toml`/`.git`), so the verbs work from any working directory — a poll
from a subdir finds the same run. `bin/mini` wraps the `mini` console-script and
pins the project, so it also runs from anywhere without an activated venv.

## The one invariant: tick vs. read

- **`run` / `retry` / `cancel` _tick_ the DAG** — they re-run `main` and launch
  (or stop) work. They have side effects and **cost money**.
- **`ls` / `status` / `watch` / `results` / `logs` only read** the durable store.
  Safe to call any time; they never relaunch.

So **poll with `status`, never by re-running.** `--app modal` inspects a run on
the Modal control plane. Don't pass `--watch` in a capped/agent session —
`run --watch` blocks to completion; one plain `run` launches the next stage and
returns at once. To follow a run with a live bar *without* driving it, use the
read-only `mini watch <name>` (renders a run another process launched — e.g. a
detached/Modal run — and never `tick`s).

**The backend sticks.** `run --app modal` stamps the launch backend into
`.mini/<name>/.app`, and every later verb without `--app` follows it — launch
with the flag once, then plain `status`/`watch`/`results`/`retry` hit the right
control plane (to move a run's home, pass `--app` again; the marker updates).
The marker is per-checkout (`.mini/` is untracked), so a fresh clone — CI, a
scheduled monitor's new environment — has no memory: set `MINI_APP=modal` in
the environment, or commit `app = "modal"` under `[tool.mini]` in
`pyproject.toml`, to make the default travel (precedence: flag > marker > env >
pyproject > local). Either way, a read that lands on an empty backend is not a
dead end: it peeks at the other one and names the flag — `no tasks found for
'pipeline' on local — found 3 task(s) on modal — try: --app modal`.

## The wake-loop

A capped session can't babysit a long run, so work in **wakes** — each verb is a
cheap, stateless call against durable state:

1. **Launch / advance:** `bin/mini run <exp>` (returns immediately).
2. **Later, poll:** `bin/mini status <exp>` (read-only).
3. **On failure:** `bin/mini logs <exp> <key>`, fix, `bin/mini retry <exp>`.
4. **When done:** `bin/mini results <exp>`, or open `report.py`.

Re-running is cheap: completed steps are memo hits, so a `run` only advances the
un-run pieces.

Watching a big sweep is cheap too: the watch loops cache settled
(`DONE`/`FAILED`/`CANCELLED`) records — they're immutable — and re-read only the
tasks still in flight, so a mostly-done sweep stops paying to poll its settled
tail (on Modal each record read is a `Dict` round-trip). Each task also records
**what it actually ran on** (host/OS/Python, and the GPU when one is attached);
`status` shows `on <GPU>` for remote tasks, and the full snapshot is on the
record under `env`.

**Queued ≠ running.** A record reads RUNNING from launch, but the worker writes
`env` as its first action — so until `env` appears, the task is *launched but
not started*. `status` shows it as `◌ queued` with its time in queue (`⧖`)
instead of a heartbeat, and `watch` tags its bar `— queued`. Locally this is a
momentary blip; on Modal a capacity-starved task can sit queued indefinitely,
and only the wall-clock budget (below) will reap it. A task stuck on `queued`
with an old `⧖` is a scheduling problem (capacity, container boot), not slow
code.

## Recovery

`FAILED` and `CANCELLED` are **terminal by design** — a plain `run` will **not**
relaunch them (a deterministic failure shouldn't busy-loop). Recover on purpose:
`bin/mini logs <exp> <key>` to read the traceback, fix, then `bin/mini retry
<exp>` (`--key <key>` for one). To re-run a `DONE` task, edit its fn or bump
`version=` — a memo hit is never silently re-run. If a re-run or memo hit
*surprises* you, `bin/mini explain <exp> <key>` shows the key's evidence and
diffs it against its sibling record (code vs. inputs, per-dependency).

The full fix loop — the fix/prune/retry table, bounded hotfixes with
`--keep-stale-done`, superseded records, and partial `map` failures — is in
[recovery.md](./recovery.md); the safety rules below are the operational side of it.

### Hotfix safety (avoid double-spending)

Editing a task fn makes every cell of it *stale*, so it re-runs (in place — keys
are identity, so nothing is orphaned). But a re-run does **not** kill workers
already detached under the old code — they keep burning (real money on Modal).
And editing a **shared helper** invalidates *every* task that calls it. Three
rules keep the blast radius bounded:

1. **Only hotfix terminal (FAILED/CANCELLED) tasks** — their worker is already
   dead, and a stale terminal task relaunches on the next `run` (no `retry`
   needed; the fix is what it was waiting for). For a transient failure, don't
   edit: `retry --key <key>` re-runs just that task (blast radius is one task).
   By default an edit also re-runs the fn's **DONE** cells — for a `map`, the
   whole fan-out. If the fix demonstrably doesn't change what the finished cells
   computed, run with `--keep-stale-done`: DONE results are served as-is (badged
   `(stale code — kept)` in `status`) and only the unfinished cells re-run.
2. **If anything is in-flight, `cancel` first, then fix.** Never edit under a
   live worker — you'd pay for compute nobody wants. (`cancel` is store-scoped —
   it also stops stale old-code workers, which keep showing as `RUNNING` in
   `status`.) The *records* are safe either way: every attempt runs under a
   generation stamp, so a stale worker that survives `cancel` can't overwrite
   its replacement's state or result — but it can still race mutable *names* it
   writes (`set_ref`, `publish`, shared volume paths).
3. **Only ever edit the single failing task fn.** Never a shared helper, `main`,
   or the DAG shape — those re-run an unbounded set of tasks. That's an
   **escalation**, not a hotfix.

`cancel` is also the cost-control lever: stop in-flight work you no longer want.

### Modal: stale serialized worker

A long-lived **detached** Modal app can keep serving a *previously serialized*
worker entry, so an edit to worker/store wiring won't take effect until its
containers drain — the run looks like it's ignoring your fix. If a worker-side
change isn't being picked up on Modal, stop the app (or launch under a fresh app
name) to force a re-serialize. (This is a property of the detached app, not the
artifact store.)

## Wall-clock budget (auto-teardown)

A detached run outlives the process that launched it, so a forgotten or wedged
run can burn money (Modal) or hold local resources **indefinitely**. Bound the
whole sweep with a wall-clock budget:

```
bin/mini run <exp> --budget 2h          # the run may not outlive 2 hours
bin/mini run <exp> --app modal --budget 30m
```

`--budget` stamps a `deadline_at` into the run's control plane at launch (a
sidecar on the same store — local JSON / Modal `Dict` — so no new infra). There's
no supervising process to fire a timer, so enforcement is **opportunistic**: any
process that already touches the store — `status`, `watch`, the `--watch`
driver — cancels in-flight tasks (→ `CANCELLED`) once the deadline passes, via the
same `cancel` path. A driver also refuses to launch a *new* stage past the
deadline. So a budgeted run that goes unattended settles cleanly the next time
anything polls it; `status` shows `budget 2h, 12m left` (or `expired`).

The budget is **run-level**, complementing the per-task `--timeout` (Modal's
function timeout, which bounds one task). Passing `--budget` again re-arms the
deadline relative to now (so you can `retry` past an expired budget); a plain
re-run to advance a multi-step DAG inherits the existing deadline. This is
distinct from `cancel` (manual, immediate) — the budget is the unattended
backstop.

## Escalation contract

Attempt only a **local, obvious** fix on a terminal task (typo, bad path, wrong
hyperparameter) within rules 1–3. Otherwise **stop and report up** — do not
guess past the mandate. Escalate when: the fix isn't local/obvious; the same
task fails again after a fix; the failure is in experiment design or `mini`
internals; or cost looks wrong (runaway relaunches). Report:

```
{ experiment, state summary, failing key(s), last error line,
  traceback excerpt, what I tried, recommended next step }
```

## Delegating & scheduling a long run

To launch and babysit a run without spending the main session's (expensive)
context, **delegate to the `experiment-monitor` subagent** (Haiku) — "poll
status of `<exp>`, advance if asked, apply a bounded hotfix if a task failed
obviously". It does one pass and reports. The Haiku monitor can't spawn other
agents, so its escalation flows back to **you**: on an escalation report, spawn
the **`experiment-doctor` subagent** (Sonnet); bring a genuine redesign to the
human rather than reshaping the experiment yourself.

For a run too long to watch in one session, set up a **scheduled routine** (the
`/schedule` skill) at a cadence the user picks — don't assume one. Each wake, the
routine spawns the monitor (and, on escalation, the doctor, then notifies). It
**self-removes when the run settles**: when `status` shows a terminal aggregate
state, find the routine's id via `CronList` (match by name) and `CronDelete` it.
A recurring cron costs money — confirm with the user before creating it.

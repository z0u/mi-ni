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

## Recovery

`FAILED` and `CANCELLED` are **terminal by design** — a plain `run` will **not**
relaunch them (a deterministic failure shouldn't busy-loop). Recover on purpose:
`bin/mini logs <exp> <key>` to read the traceback, fix, then `bin/mini retry
<exp>` (`--key <key>` for one). To re-run a `DONE` task, edit its fn or bump
`version=` — a memo hit is never silently re-run.

### Hotfix safety (avoid double-spending)

Editing a task fn changes its memo key, so it re-runs. But a re-run does **not**
kill workers already detached under the *old* key — they keep burning (real
money on Modal). And editing a **shared helper** invalidates *every* task that
calls it. Three rules keep the blast radius bounded:

1. **Only hotfix terminal (FAILED/CANCELLED) tasks** — their worker is already
   dead. Fix the fn, then `retry --key <key>`; blast radius is one task.
2. **If anything is in-flight, `cancel` first, then fix.** Never edit under a
   live worker. (`cancel` is store-scoped — it also stops orphaned old-version
   workers, which keep showing as `RUNNING` in `status`.)
3. **Only ever edit the single failing task fn.** Never a shared helper, `main`,
   or the DAG shape — those re-run an unbounded set of tasks. That's an
   **escalation**, not a hotfix.

`cancel` is also the cost-control lever: stop in-flight work you no longer want.

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

# Autonomous experiments: what mi-ni would need

_Exploration, not a committed plan. Companion proof-of-concept: [`agentic_poc.py`](./agentic_poc.py)._

The goal: an agent (Claude Code on the web) takes an experiment description,
writes the code, launches runs, monitors them, fixes problems, and reports
results — without a human in the loop. The web harness caps process runtime, so
the agent can't babysit a run with a long-lived monitor. It works in **wakes**:
launch and stop; later, wake, check, act; repeat. mi-ni has to make each wake a
cheap, stateless-from-the-agent's-view call against durable state.

## Diagnosis: the lifecycle is fused to one process

`Apparatus.amap` does launch + monitor + collect in a single blocking generator,
all scoped to `async with self.app.run()` (see `modal_apparatus.py:185`). When
that process ends, almost everything ends with it:

| Concern | Today | Survives process death? |
| --- | --- | --- |
| Remote work | `modal_fn.map.aio(...)` inside `app.run()` | **No** — app torn down on exit |
| Progress | `modal.Queue.ephemeral()` → Rich display | **No** — ephemeral, in-process consumer |
| Results | yielded in-process, held in a cell variable | **No** |
| Errors | raised into the caller; traceback to console | **No** |
| Artifacts | written to the named Volume, committed at job end | **Yes** |

Only the Volume is durable. So the agent-shaped change is not really "detach
Modal" — that keeps the *work* alive but leaves it a black box. The 80% is
**externalizing progress, results, and errors to durable storage and making them
pollable**. Detaching is the other 20%.

## 1. What changes in `src/mini/`

Split the lifecycle into phases that can run in **different processes**, backed
by durable state:

- **`submit`** — launch detached, persist a manifest, return a handle. Non-blocking.
- **`status`** — read durable per-job state. Fast, stateless, idempotent.
- **`gather`** — collect results/errors once settled.
- **`cancel`** — tear down (cost control).

Concretely:

- **A `RunStore` on top of the existing `Volume`.** Manifest, per-job
  `status.json`, `result.pkl`, `error.txt`. Both backends share it — the Volume
  is the one thing that already survives, and building on it keeps local and
  Modal identical. The PoC uses a plain dir as the store and it Just Works.
- **A durable, pluggable progress sink.** `emit_progress`/`QueueLike` are already
  sink-agnostic (the PoC swaps in a `FileStatusSink` with zero changes to
  experiment code). Add a Volume-backed and a named-`modal.Dict`-backed sink.
  The Rich display becomes one *reader* among others, used only for interactive
  `map`. Split by latency: **progress → named `modal.Dict`** (live, cheap, no
  commit); **results/errors/artifacts → Volume** (committed).
- **Persist results, not just return them.** The Modal wrapper already commits
  the Volume (`modal_apparatus.py:288`); also write `result.pkl` / `error.txt` /
  `status.json` per job so `gather` works from any process.
- **Capture failures durably.** Wrap `fn` in try/except: write the full
  traceback + set state `FAILED`. Route per-job logs to the Volume (the
  `before_each(logging_config.apply)` hook is the seam). "Fix problems" is
  impossible if the only copy of the traceback died with the launcher.
- **Detached Modal execution.** Swap `fn.map.aio()` for `fn.spawn`/`spawn_map`
  under `app.run(detach=True)` (or a deployed app); persist the `FunctionCall`
  IDs in the manifest; reconnect with `FunctionCall.from_id(...)`. All present in
  Modal 1.5. Use `FunctionCall.get(timeout=0)` as a liveness/exception fast-path
  (catch the container that OOM'd between wakes).
- **Heartbeats + liveness.** Status carries a last-heartbeat so the agent can
  tell "running" from "silently died" (timeout/OOM) — a queue that simply went
  quiet is ambiguous today.
- **Idempotent, resumable submit.** Key jobs by a hash of their config; a
  re-`submit` skips settled jobs. Makes the wake loop safe to call repeatedly and
  lets the agent relaunch only the failed subset after a fix.

## 2. Experiment structure: separate definition from report

Not "scripts instead of notebooks" — **demote the notebook from execution engine
to report.** Two reasons the reactive notebook can't be the execution surface:
its training cell *blocks* holding results in memory, and later cells consume
that memory. That is a long-lived-session model by construction.

Split each experiment into:

- **Definition — plain, importable module.** Sweep configs + the job function
  (e.g. `train_one`), importable with no Marimo session and no UI state. The
  remote workers and the agent both import it. The heavy lifting already lives in
  `experiment/`; what must move out of notebook cells is the **orchestration**
  (the sweep definition and the `amap` call).
- **Report — the notebook (or generated Markdown/PNG).** Reads *durable results*
  from the `RunStore` and renders. This also fixes a current annoyance: reopening
  `gpt_sweep.py` re-runs the whole sweep to see its plots; reading persisted
  results means the report renders standalone.

```
experiments/arch-sweep/
  experiment.py   # sweep configs + train_one  (importable, no UI)
  report.py       # marimo notebook: loads results for a run id, plots
```

Keep notebooks — literate narrative is a stated project value. Just stop making
them the thing that holds the run.

## 3. Concrete API

A durable, id-addressable `Run` handle, plus lifecycle verbs:

```py
class RunState(StrEnum): PENDING; RUNNING; DONE; FAILED; CANCELLED

@dataclass
class JobStatus:
    job_id: str; state: RunState
    step: int; total: int; message: str
    error: str | None          # traceback if FAILED
    heartbeat_at: float | None

class Apparatus:
    def submit(self, fn, *iterables, kwargs=None) -> Run: ...   # non-blocking, detached
    def reopen(self, run_id: str) -> Run: ...                   # reconstruct in a fresh process

class Run:
    id: str
    def status(self) -> list[JobStatus]: ...   # poll; reads durable truth
    def done(self) -> bool: ...
    def results(self, wait=False) -> list[R]: ...
    def cancel(self) -> None: ...
    def logs(self, job_id: str) -> str: ...
```

`map`/`amap` stay, redefined as `submit` + blocking poll + `gather`, so
interactive/local use is unchanged. Detached mode is opt-in:
`ModalApparatus(...).w(detached=True)` or `.detached()`.

The agent's loop across wakes:

```py
# wake 1 — launch
run = ModalApparatus('arch-sweep').w(gpu='L4', detached=True).submit(train_one, configs)
print(run.id)                 # agent records this, then ends its turn

# wake N — fresh process
run = ModalApparatus('arch-sweep').reopen(run_id)
st = run.status()
if any(j.state == 'FAILED' for j in st):
    ...                       # read j.error, edit code, re-submit failed subset
elif run.done():
    write_report(run.results())
# else: still running — end turn, check again next wake
```

A thin CLI (`experiment.py launch|status|gather|cancel`) wires the sweep to these
verbs so the agent drives it without holding a session. **Waking itself is the
harness's job** (scheduled self-check / cron / PR webhook), not mini's — mini's
job is to make `status()` a cheap, honest read.

## What the PoC validated (local compute only)

`agentic_poc.py` splits the lifecycle into `launch` / `poll` / `gather` CLI
invocations. Observed:

- Launcher spawned detached workers and exited; **7 separate poll processes**
  read live, advancing progress with the launcher long dead.
- An injected crash showed up as `FAILED` + traceback tail to every later poll
  and to `gather` — the diagnose-and-fix signal is durable.
- A real eventual-consistency wrinkle: a poll read a job as `RUNNING` while
  `gather` a moment later found its result. Lesson baked into the API above:
  `status`/`gather` always read durable truth, never trust the last poll.

## Deferred / open

- `app.run(detach=True)` (ephemeral, persist FunctionCall IDs) vs `app.deploy()`
  (named, robust across sessions, needs a deploy step). Lean ephemeral first.
- Auto-teardown on a wall-clock budget so a forgotten detached run can't burn
  money indefinitely.
- Garbage-collecting old run dirs / named Dicts.

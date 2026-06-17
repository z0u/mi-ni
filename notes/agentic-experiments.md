# Autonomous experiments: what mi-ni would need

_Design + status. The **local backend is implemented**, including memoized
multi-step orchestration: `mini.runs` (control plane, `Run`, discovery),
`mini.memo` (content-addressed fingerprint + store), `mini.orchestration`
(`Ctx`/`Pending`/`tick`), `mini.experiment`, the detached workers
(`mini._worker`, `mini._taskworker`), and the `python -m mini` CLI
(`run`/`launch`/`status`/…) — exercised by `experiments/{toy,pipeline}.py` and
`tests/mini/test_{runs,orchestration}.py`. The Modal backend (detached
`spawn*map`+`modal.Dict` control plane / task spawning) is still to do.
Original local-only proof-of-concept: [`agentic_poc.py`](./agentic_poc.py).*

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

<!-- prettier-ignore -->
| Concern | Today | Survives process death? |
| --- | --- | --- |
| Remote work | `modal_fn.map.aio(...)` inside `app.run()` | **No** — app torn down on exit |
| Progress | `modal.Queue.ephemeral()` → Rich display | **No** — ephemeral, in-process consumer |
| Results | yielded in-process, held in a cell variable | **No** |
| Errors | raised into the caller; traceback to console | **No** |
| Artifacts | written to the named Volume, committed at job end | **Yes** |

Only the Volume is durable. So the agent-shaped change is not really "detach
Modal" — that keeps the _work_ alive but leaves it a black box. The 80% is
**externalizing progress, results, and errors to durable storage and making them
pollable**. Detaching is the other 20%.

## Two planes: control vs I/O

The single most clarifying decision: split durable state into two planes with
different access patterns, rather than one `RunStore`.

- **Control plane** — small, hot, last-writer-wins: per-job state, step/total,
  heartbeat, latest metrics, the `FunctionCall` ids, a run index. On Modal this
  is a **named `modal.Dict`** (Redis-backed, read/written from the _client_ with
  no remote function and no `commit`/`reload`). Locally it's a JSON file/dir.
- **I/O plane** — large, cold, append-mostly: datasets, weights, big result
  objects, full logs/tracebacks. This is the existing **Volume**.

Keeping them separate is conceptually right and operationally necessary: polling
status must not require spinning up a remote function or committing a Volume, and
bulk artifacts must not bloat a Redis dict.

## 1. What changes in `src/mini/`

Split the lifecycle into phases that can run in **different processes**, backed
by the two planes:

- **`submit`** — launch detached, write the run to the control plane, return a handle. Non-blocking.
- **`status`** — read the control plane. Fast, stateless, idempotent.
- **`gather`** — collect results from the I/O plane once settled.
- **`cancel`** — tear down (cost control).

Concretely:

- **A `ControlPlane`, addressed by experiment _name_.** Backends: `modal.Dict`
  (Modal) / a local dir. Holds a run index + per-job records. Because it's keyed
  by the stable experiment name — not an ephemeral run id — a fresh process can
  _discover_ runs it never launched (see Discovery, below).
- **Retire the queue from the detached path.** A queue is push-based and
  _accumulates_: abandon a run for days and a named queue is full or gone.
  Progress is last-writer-wins — you want the _current_ step, not every
  intermediate — so detached comms are state writes to the control plane, which
  can't fill. `emit_progress`/`QueueLike` are already sink-agnostic (the PoC
  swaps in a `FileStatusSink` with zero changes to experiment code), so this is a
  sink swap, not a rewrite. Keep `LocalQueue` only for the interactive `map` live
  display — or retire it too and have the Rich display poll the control plane.
- **Metrics in the status.** `JobStatus` carries `metrics: dict[str, float]`
  (latest scalars — loss, grad-norm, lr). This is what lets the agent or a human
  spot a run going sideways and intervene. Latest scalars live in the control
  plane (cheap to poll); full history goes to the I/O plane (or wandb — already a
  dep, but lean on the self-contained control plane for autonomy).
- **Persist results, not just return them.** The Modal wrapper already commits
  the Volume (`modal_apparatus.py:288`); also write `result.pkl` / `error.txt`
  per job so `gather` works from any process.
- **Capture failures durably.** Wrap `fn` in try/except: write the full
  traceback to the I/O plane + set state `FAILED` in the control plane. Route
  per-job logs to the Volume (the `before_each(logging_config.apply)` hook is the
  seam). "Fix problems" is impossible if the only copy of the traceback died with
  the launcher.
- **Detached Modal execution (verified against Modal 1.5 source).** Swap
  `fn.map.aio()` for `fn.spawn_map` under `app.run(detach=True)`. Confirmed:
  `detach=True` sets `APP_STATE_DETACHED` and the app keeps running after the
  client disconnects (`runner.py`); `spawn` needs no _deployed_ function (its only
  guard is "no web endpoints"); `spawn_map` returns **one** `FunctionCall` for the
  whole sweep, and `FunctionCall.from_id(fc_id).get(index=i, timeout=0)`
  reconnects and polls a single job from a fresh process with **no app context**.
  So: persist the one `fc_id` in the control plane; poll per index. Deploy is an
  optimization (a stable named function to spawn against repeatedly), not a
  requirement — default to detached-ephemeral.
- **Treat `FunctionCall.get` as liveness, not storage.** Modal retains call
  results only for a server-side window, and `.get()` re-raises the remote
  exception. Use it to detect crashes/OOM between wakes; use the Volume as the
  durable source of truth for `gather`.
- **Heartbeats + liveness.** Status carries a last-heartbeat so the agent can
  tell "running" from "silently died" (timeout/OOM), cross-checked against
  `FunctionCall.get(timeout=0)`. Locally, a pid/process-group check.
- **Idempotent, resumable submit.** Key jobs by a hash of their config; a
  re-`submit` skips settled jobs. Makes the wake loop safe to call repeatedly and
  lets the agent relaunch only the failed subset after a fix.

## 2. Experiment structure: separate definition from report

Not "scripts instead of notebooks" — **demote the notebook from execution engine
to report.** Two reasons the reactive notebook can't be the execution surface:
its training cell _blocks_ holding results in memory, and later cells consume
that memory. That is a long-lived-session model by construction.

Split each experiment into:

- **Definition — plain, importable module.** Sweep configs + the job function
  (e.g. `train_one`), importable with no Marimo session and no UI state. The
  remote workers and the agent both import it. The heavy lifting already lives in
  `experiment/`; what must move out of notebook cells is the **orchestration**
  (the sweep definition and the `amap` call).
- **Report — the notebook (or generated Markdown/PNG).** Reads _durable results_
  from the planes and renders. This also fixes a current annoyance: reopening
  `gpt_sweep.py` re-runs the whole sweep to see its plots; reading persisted
  results means the report renders standalone.

```
experiments/arch-sweep/
  experiment.py   # Experiment(name, apparatus, fn, configs)  — importable, no UI
  report.py       # marimo notebook: loads results for a run id, plots
```

Keep notebooks — literate narrative is a stated project value. Just stop making
them the thing that holds the run. The directory convention belongs in an Agent
Skill (sibling to `mi-ni`) plus a worked `docs/` example, so future agents
structure new experiments without re-deriving this.

## 3. Concrete API

A durable, id-addressable `Run` handle, plus lifecycle verbs:

```py
class RunState(StrEnum): PENDING; RUNNING; DONE; FAILED; CANCELLED

@dataclass
class JobStatus:
    job_id: str; state: RunState
    step: int; total: int; message: str
    metrics: dict[str, float]   # latest scalars — loss, grad-norm, lr
    error: str | None           # traceback if FAILED
    heartbeat_at: float | None

class Apparatus:
    def submit(self, fn, *iterables, kwargs=None) -> Run: ...   # non-blocking, detached
    def reopen(self, run_id: str) -> Run: ...                   # reconstruct in a fresh process

class Run:
    id: str
    def status(self) -> list[JobStatus]: ...   # poll the control plane
    def done(self) -> bool: ...
    def results(self, wait=False) -> list[R]: ...   # from the I/O plane
    def retry(self, failed_only=True) -> Run: ...
    def cancel(self) -> None: ...
    def logs(self, job_id: str) -> str: ...
```

`map`/`amap` stay, redefined as `submit` + blocking poll + `gather`, so
interactive/local use is unchanged. Detached mode is opt-in:
`ModalApparatus(...).w(detached=True)` or `.detached()`.

### Discovery: finding a run without its id

The payoff of the name-addressed control plane. `modal.Dict.from_name(
f'mini-cp-{experiment}', create_if_missing=True)` opens from any process knowing
only the experiment name. Inside, a run index makes runs enumerable:

```
mini-cp-arch-sweep            # modal.Dict, addressed by experiment NAME
  index            -> {run_id: {state, created_at, n_jobs}, ...}
  <run_id>/job/<j> -> {state, step, total, heartbeat_at, metrics, fc_id}
```

So `status('arch-sweep')` with no run id reads `index`, finds the active/latest
run, and reports it. This mirrors the hierarchy **Experiment (name) → Run (id) →
Job (id)**, with the name as the discovery key. Liveness = heartbeat freshness
cross-checked with `FunctionCall.get(timeout=0)`.

### CLI: `python -m mini` (`src/mini/__main__.py`)

The agent drives short-lived processes; each reads/writes the control plane:

```bash
python -m mini launch experiments/arch_sweep.py   # → arch-sweep/3f9a  (detached)
python -m mini ls                                  # experiments + active runs
python -m mini status arch-sweep                   # latest run by NAME: state + metrics
python -m mini status arch-sweep/3f9a              # a specific run
python -m mini logs   arch-sweep/3f9a --job 2      # full traceback from the I/O plane
python -m mini retry  arch-sweep/3f9a --failed     # re-submit only failed jobs
python -m mini cancel arch-sweep/3f9a
python -m mini results arch-sweep/3f9a             # gather
python -m mini report  arch-sweep/3f9a             # render report.py → HTML
```

`__main__.py` is a thin shell over the `Run` API; `load_experiment(path)` imports
the module and expects a module-level `experiment = Experiment(name, apparatus,
fn, configs)`. **Waking is the harness's job** (scheduled self-check / cron / PR
webhook), not mini's — mini's job is to make `status` a cheap, honest read.

## 4. Composition across detached steps

Multi-step experiments — `ys = run(f, xs); zs = run(g, ys)` — need care once
detached, because `g` may not even be defined when `f` launches, and the process
holding `ys` is gone before `g` runs.

- **Steps need not share a Modal App.** They already don't: each `amap` opens its
  own ephemeral `app.run()` today. `f` and `g` share only the App _name_.
- **The App stops being the composition boundary; the I/O plane becomes it.** A
  detached step's return value can't be handed to the next step in memory — so
  `f` writes `ys` to the Volume (+ a summary to the control plane), and a later,
  separately-submitted `g` reads `ys` from the Volume. No shared App, no
  co-definition, no deploy.
- Deploy wouldn't help here anyway: a deployed app needs `f` _and_ `g`
  co-defined at deploy time, which contradicts "`g` defined after `f`'s results."
  Detached-ephemeral + Volume is strictly simpler for the incremental flow.

## 5. Multi-step experiments: memoized orchestration

Section 4's "write to the Volume, read it in the next step" is correct but
manual. The `gpt_sweep.py` shape — one `prepare_data` step whose output feeds a
training sweep — wants that hand-off to be automatic. The single-map
`Experiment(name, fn, configs)` can't express it (the configs depend on prep's
return value). Validated alternative (see [`multistep_poc.py`](./multistep_poc.py)):

**The experiment is an orchestration function, `main(ctx)` — a plain-Python DAG —
and every `ctx.run`/`ctx.map` is content-addressed (memoized).** Each call:
cached → return the stored result; in flight → suspend; absent/failed → launch a
detached worker, then suspend. The driver re-runs `main` every wake; completed
steps are memo hits, so only the un-run / failed pieces execute. **Crash-recovery
== re-run the whole thing**, exactly as proposed.

```py
def main(ctx):
    meta = ctx.run(prepare_data)                       # single step
    configs = derive_configs(meta)                     # ordinary Python (cheap, deterministic)
    return ctx.map(train_one, configs)                 # sweep that depends on prep
```

This subsumes the current model: a plain sweep is just `main = lambda ctx:
ctx.map(train, configs)`. It reuses everything already built — the control plane,
the detached worker, `Run` — with two additions: a `Ctx` that memoizes, and a
content key that replaces the random run id.

The PoC ran the gpt_sweep shape across 10 wakes: prep launched and the wake
suspended until it finished (touching `meta` suspends); the sweep then launched
all three jobs; an injected transient crash in one job healed on a later wake
(re-run relaunched only it — attempt counts were 1/2/1); a final re-run was an
instant memo hit. prep executed exactly once.

Design points (as implemented in `mini.memo` / `mini.orchestration`):

- **The key is a dependency-aware _source_ fingerprint, not cloudpickle.**
  Tempting to hash `cloudpickle.dumps(fn)` (it captures by-value deps), but
  measured fact: **its bytes differ across processes**, so it would miss the
  cache on every wake (each wake is a fresh process). Instead `mini.memo`
  fingerprints `inspect.getsource(fn)` _plus the source of the project
  functions/classes it references, transitively_ — deterministic, and it
  captures changes in your own helpers (which a bare `getsource(fn)` would miss).
  Site-packages and the mini framework are excluded, so library churn doesn't
  bust the cache. `version=` is an explicit override. Pure input-keying was
  rejected: it returns the stale buggy result after a fix — the opposite of what
  the loop needs.
- **Results live in the I/O plane; the memo record holds state/metrics only.**
- **The orchestration body re-runs every wake**, so code _between_ `ctx` calls
  must be cheap and deterministic (deriving configs, not training). Heavy or
  non-deterministic work belongs inside a task, with its seed folded into the
  inputs so the memo is honest.
- **Suspension:** `ctx.run` raises `Pending` the moment a result isn't ready;
  `ctx.map` launches _all_ missing tasks first, then suspends (so a sweep
  parallelises). Independent single steps serialise across wakes — use `map` for
  fan-out.
- **CLI:** `python -m mini run experiment.py` = one tick of `main` (advance +
  report the memo graph); incremental sweeps fall out for free (adding a config
  runs only the new key); `mini launch` remains for the one-shot run/job model.

**Guidance for the agent skill (to write later):** the memo key is
`fingerprint(fn) + inputs`, so _cache hits are maximised by passing each task the
minimal subset of config it actually uses_. A `train` keyed on the entire
experiment config re-runs whenever any unrelated field changes; keyed on
`(lr, vocab_size)` it only re-runs when those change. The skill should teach:
pass narrow inputs; keep `main` cheap/deterministic; fold RNG seeds into inputs;
bump `version=` (or just edit the function) to force a re-run.

## What the PoCs validated (local compute only)

`agentic_poc.py` — the single-sweep lifecycle as `launch` / `poll` / `gather`:

- Launcher spawned detached workers and exited; **7 separate poll processes**
  read live, advancing progress with the launcher long dead.
- An injected crash showed up as `FAILED` + traceback tail to every later poll
  and to `gather` — the diagnose-and-fix signal is durable.
- A real eventual-consistency wrinkle: a poll read a job as `RUNNING` while
  `gather` a moment later found its result. Lesson baked into the API above:
  `status`/`gather` always read durable truth, never trust the last poll.

`multistep_poc.py` — the memoized multi-step model (section 5): a prep→sweep DAG
resumed across wakes, prep memoized to a single execution, and a crashed job
healed by re-run without re-running its siblings.

## Deferred / open

- **Resolved:** detached-ephemeral (`app.run(detach=True)` + persisted
  `FunctionCall` ids) is the default; `app.deploy()` is a later optimization for a
  stable function spawned against repeatedly.
- Auto-teardown on a wall-clock budget so a forgotten detached run can't burn
  money indefinitely.
- Garbage-collecting old run records / named Dicts / Volume run dirs.
- Whether to retire `LocalQueue` entirely (display polls the control plane) or
  keep it for the interactive path only.

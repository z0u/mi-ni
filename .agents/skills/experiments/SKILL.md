---
name: experiments
description: How to define, run, and monitor experiments with mi-ni's memoized orchestration — the durable, detached flow that survives process death, so an agent (or a human) can drive, poll, fix, and report on long or multi-step experiments from the CLI.
---

# Running experiments

mi-ni has **two ways to compute**:

- **Interactive `Apparatus`** — a blocking call for quick, light work you watch
  finish. Dies with the process. See the `mi-ni` skill.
- **Memoized orchestration** (this skill) — work is launched detached and its
  state, results, and errors are written to a durable store. You drive and poll
  it from short-lived CLI processes. Use it for sweeps, multi-step pipelines,
  anything slow, and anything an agent runs autonomously.

## The model: `main(ctx)` is a DAG

An experiment is a plain function `main(ctx)` that expresses the dependency graph
in ordinary Python. Each `ctx.run`/`ctx.map` is **content-addressed (memoized)**:

- cached → returns the stored result immediately;
- in flight → **suspends** the wake (raises `Pending`);
- absent / reset → launches a detached worker, then suspends.

A driver re-runs `main` on every _wake_. Completed steps are memo hits, so only
the un-run pieces execute. Crash recovery is just "run it again."

```python
from mini import Ctx, Experiment

def main(ctx: Ctx) -> dict:
    meta = ctx.run(prepare_data)                      # one step; suspends until done
    configs = [(lr, meta['vocab_size']) for lr in LRS]  # plain Python between steps
    return ctx.map(train, configs)                    # fan-out that depends on prep

experiment = Experiment(name='my-exp', main=main)
```

The module exposes a top-level `experiment = Experiment(...)`. It carries **no
compute** — the apparatus is injected when it runs (see _Choosing compute_), so
the same file runs locally or on Modal without edits.

## Where experiments live

```
**/<name>/
  experiment.py   # the definition: main(ctx); importable, no UI.
  report.py       # a Marimo notebook that READS durable results and renders them. Published.
```

Split definition from report. The definition is imported by the CLI and the
remote workers; the report reads persisted results and plots, so it opens
standalone without re-running the work. See `docs/pipeline/` for a worked,
runnable example.

## Driving vs. reading

`mini run`/`retry` **tick** the DAG: they re-run `main` and _launch_ missing or
retryable work. They have side effects. Everything else only reads the durable
store.

```bash
./bin/mini run foo/experiment.py --watch   # drive to completion, live bar
./bin/mini run foo/experiment.py           # advance one wake, return at once
./bin/mini ls                 # experiments + rolled-up state
./bin/mini status foo         # per-task state + latest metrics (read-only)
./bin/mini results foo        # per-task results (read-only)
./bin/mini logs   foo <key>   # a failed task's full traceback
./bin/mini retry  foo         # reset FAILED/CANCELLED, then advance
./bin/mini cancel foo         # stop in-flight tasks (cost control)
```

State is addressed by experiment name (one store per experiment). All verbs take
`--app modal` to inspect a run on the Modal control plane. `--watch` drives to
completion with a live bar; Ctrl-C stops only the _watch_ — detached workers
live on, so re-running resumes.

## Memoization: write cache-friendly experiments

The key is `fingerprint(source of fn + the project fns it calls) + inputs`. So:

- **Pass each task the narrow subset of config it actually uses.** `train(lr,
vocab_size)` re-runs only when `lr` or `vocab_size` change; `train(whole_config)`
  re-runs whenever _any_ unrelated field changes.
- **Keep `main` cheap and deterministic** — it re-runs every wake. Derive configs
  there; do heavy or random work _inside_ a task.
- **Fold RNG seeds into the inputs**, so the memo is honest (same inputs ⇒ same
  result). A task seeded from wall-clock can never be a cache hit.
- **Force a re-run** by editing the function (the source fingerprint changes) or
  passing `version='v2'`. Editing a project helper a task calls also invalidates it;
  library/framework churn does not.

See [references/memoization.md](references/memoization.md) for the fix / prune /
retry semantics in full, with worked examples.

## Fix, prune, retry (the recovery loop)

- **Fix a bug → re-run.** Edit the task fn and `bin/mini run …`; the changed key
  re-runs, untouched siblings stay memo hits.
- **Prune an item** — remove it from `configs`. It's simply not requested; the
  rest are hits.
- **A `FAILED`/`CANCELLED` task is terminal** — `run` won't relaunch it. Recover
  on purpose: `bin/mini logs <name> <key>` to read the traceback, fix, then
  `bin/mini retry <name>` (optionally `--key <key>` for just one).
- **Re-run a `DONE` task** — edit its fn or bump `version=`; a memo hit is never
  silently re-run.

## Choosing compute

Compute is an execution choice, not part of the definition.

- **CLI:** `--app local` (default) / `--app modal`; `--workers N` sets local
  concurrency. `mini` builds the apparatus and injects it.
- **Per-step, from a notebook:** `ctx.run(fn, on=cpu)` / `ctx.map(fn, items,
on=gpu)` routes individual steps to different apparatuses (e.g. CPU prep, GPU
  training). Each step also picks up that apparatus's `before_each` hooks.

## The agent wake-loop

The web harness caps process runtime, so an agent can't babysit a long run. It
works in **wakes**: launch, stop; later wake, check, act; repeat. This whole flow
is built for that — each verb is a cheap, stateless call against durable state:

1. **Launch / advance:** `bin/mini run <exp>` (don't block on `--watch` in a
   capped session — one tick launches the next stage and returns).
2. **Later, poll:** `bin/mini status <exp>` — read-only; never relaunches.
3. **On failure:** `bin/mini logs <exp> <key>`, fix the code, `bin/mini retry <exp>`.
4. **When done:** `bin/mini results <exp>`, or open `report.py`.

Waking is the harness's job (a scheduled self-check, cron, or PR webhook), not
mini's — mini's job is to make each wake honest and cheap.

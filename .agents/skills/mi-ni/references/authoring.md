# Authoring an experiment

An experiment is a plain function `main(ctx)` that expresses a dependency graph
in ordinary Python. Each `ctx.run`/`ctx.map` call is **content-addressed
(memoized)**: a cache hit returns the stored result; in-flight work suspends the
wake (raises `Pending`); absent work launches a detached worker, then suspends.
A driver re-runs `main` on every _wake_, so completed steps are memo hits and
only the un-run pieces execute. Crash recovery is just "run it again".

```python
from mini import Ctx, Experiment

def main(ctx: Ctx) -> dict:
    meta = ctx.run(prepare_data)                        # one step; suspends until done
    vocab = meta["vocab_size"]                          # plain Python between steps
    return ctx.map(train, LRS, [vocab] * len(LRS))      # fan-out that depends on prep

experiment = Experiment(name="my-exp", main=main)
```

The module exposes a top-level `experiment = Experiment(...)`. It carries **no
compute** — the apparatus is injected when it runs, so the same file runs
locally or on Modal without edits.

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

## Write cache-friendly experiments

The memo key is the task's *identity* — `fn name + fingerprint(inputs)` — and
each attempt carries *evidence* (`fingerprint(source of fn + the project fns it
calls)`) that decides whether its cached result is still current; stale evidence
re-runs the task in place, under the same key (full semantics in
[memoization.md](./memoization.md)). To keep the "fix a bug, re-run" loop fast
and honest:

- **Pass each task the narrow subset of config it actually uses.** `train(lr,
  vocab_size)` re-runs only when `lr` or `vocab_size` change; `train(whole_config)`
  re-runs whenever _any_ unrelated field changes.
- **Keep `main` cheap and deterministic** — it re-runs every wake. Derive configs
  there; do heavy or random work _inside_ a task.
- **Fold RNG seeds into the inputs**, so the memo is honest (same inputs ⇒ same
  result). A task seeded from wall-clock can never be a cache hit.
- **Force a re-run** by editing the function (its evidence goes stale) or passing
  `version="v2"` — either way a new attempt on the same record. Editing a project
  helper a task calls also invalidates it; library/framework churn does not.

## Returning large outputs

A step's result holds the *small* thing (metrics, a handle). For *large* bytes —
an activation cache, an eval dump, a figure — don't return a volume `Path` (it
pickles a location that may evaporate). `put` them into the content-addressed
store and return the `Artifact` handle instead:

```python
from mini.store import put

def extract(cfg) -> dict:
    art = put(get_data_dir() / "acts", name="activations")  # handle, not a path
    return {"cfg": cfg.id, "activations": art}
```

The store is **project-scoped**, so one experiment can hand an artifact to
another by name (`set_ref`/`get_ref`) with no recompute. Full semantics — trees,
publishing artifacts to a URL, the Modal caveat — in [storage.md](./storage.md).
Externalizing a *report's* figures and data for publishing is in
[reports.md](./reports.md); the `themed` hook for it is in
[vis.md](./vis.md).

## Routing steps to compute

Compute is an execution choice, not part of the definition. A file experiment
stays backend-agnostic by tagging steps with a **role** that the CLI/driver maps
to a concrete apparatus:

```python
meta = ctx.run(prepare_data, role="cpu")          # prep on CPU
return ctx.map(train, configs, role="gpu")        # training on GPU (fn(config) per item)
```

Each step also picks up that apparatus's `before_each` hooks. The default role
is the tick's apparatus (set by `--app` / `--workers` on the CLI). In a
notebook, where you already hold apparatus handles, you can instead pass an
instance directly: `ctx.run(fn, on=cpu_app)`.

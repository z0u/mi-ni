# Authoring an experiment

An experiment is a plain function `main(ctx)` that expresses a dependency graph in ordinary Python. Each `ctx.run`/`ctx.map` call is _content-addressed_ (memoized): a cache hit returns the stored result; in-flight work suspends the wake (raises `Pending`); absent work launches a detached worker, then suspends. A driver re-runs `main` on every _wake_, so completed steps are memo hits and only the un-run pieces execute. Crash recovery is just "run it again".

```python
from mini import Ctx, Experiment

def main(ctx: Ctx) -> dict:
    meta = ctx.run(prepare_data)                        # one step; suspends until done
    vocab = meta["vocab_size"]                          # plain Python between steps
    return ctx.map(train, LRS, [vocab] * len(LRS))      # fan-out that depends on prep

experiment = Experiment(name="my-exp", main=main)
```

`ctx.map` zips its iterables Executor-style — `train(lr, vocab)` per pair, and mismatched lengths raise — which is why a constant is repeated to length above. A single iterable passes each element as one argument, tuples included.

The module exposes a top-level `experiment = Experiment(...)`. It carries no compute: the apparatus is injected when it runs, so the same file runs locally or on Modal without edits.

## Where experiments live

```
**/<name>/
  experiment.py   # the definition: main(ctx); importable, no UI.
  report.py       # a Marimo notebook that READS durable results and renders them. Published.
```

Split definition from report. The definition is imported by the CLI and the remote workers; the report reads persisted results and plots, so it opens standalone without re-running the work. See `docs/probe/` for a worked, runnable example with both halves.

Two conventions that keep the pairing healthy:

- Avoid naming a `src/` package after a docs filename. Marimo runs a report with its own directory first on `sys.path`, so the sibling `experiment.py` shadows any package named `experiment`. Symptom: import dies with "'X' is not a package".
- Extract shared testbed code. When a new experiment starts by copying a sibling's model/eval helpers, consider lifting them into a shared `src/` package (unless you expect heavy churn). Memoization evidence tracks project source transitively, so the split doesn't weaken cache correctness.

CI globs `docs/**/experiment.py` (`tests/mini/test_experiments_e2e.py`): every definition is at least *loaded* (import + construct), and the light demos run to completion. So a new or renamed experiment gets rot coverage for free — but its module top level must stay cheap and side-effect-free (imports of heavy deps belong inside task fns, which also keeps the driver light).

## Write cache-friendly experiments

The memo key is the task's *identity* — `fn name + fingerprint(inputs)` — and each attempt carries *evidence* (`fingerprint(source of fn + the project fns it calls)`) that decides whether its cached result is still current; stale evidence re-runs the task in place, under the same key (full semantics in [memoization.md](./memoization.md)). To keep the "fix a bug, re-run" loop fast and correct:

- Pass each task the narrow subset of config it actually uses — the habit that pays best. `train(lr, vocab_size)` re-runs only when `lr` or `vocab_size` change; `train(whole_config)` re-runs whenever _any_ unrelated field changes.
- Keep `main` cheap and deterministic; it re-runs every wake. Derive configs there; do heavy or random work _inside_ a task.
- Fold RNG seeds into the inputs, so the same inputs really do produce the same result. A task seeded from wall-clock can never be a cache hit.
- Force a re-run by editing the function (its evidence goes stale) or passing `version="v2"` — either way a new attempt on the same record. Editing a project helper a task calls also invalidates it; library/framework churn does not.
- Put a new helper wherever it belongs. Evidence is traced per definition, so a function added to an existing module is invisible to every task that doesn't call it; splitting it into a fresh module to protect the cache achieves nothing, and moving an *existing* helper re-runs its callers. The file-sized exceptions (package `__init__.py`, classes) are in [memoization.md](./memoization.md).

## Returning large outputs

A step's result holds the *small* thing (metrics, a handle). For *large* bytes — an activation cache, an eval dump, a figure — don't return a volume `Path` (it pickles a location that may evaporate). `put` them into the content-addressed store and return the `Artifact` handle instead:

```python
from mini.store import put

def extract(cfg) -> dict:
    art = put(get_data_dir() / "acts", name="activations")  # handle, not a path
    return {"cfg": cfg.id, "activations": art}
```

The store is project-scoped, so one experiment can hand an artifact to another by name (`set_ref`/`get_ref`) with no recompute. Full semantics (trees, publishing artifacts to a URL, the Modal caveat) are in [storage.md](./storage.md). Externalizing a *report's* figures and data for publishing is in [reports.md](./reports.md); the `themed` hook for it is in [vis.md](./vis.md).

## Routing steps to compute

Compute is an execution choice, not part of the definition. A file experiment stays backend-agnostic by tagging steps with a _role_ that the CLI/driver maps to a concrete apparatus:

```python
meta = ctx.run(prepare_data, role="cpu")          # prep on CPU
return ctx.map(train, configs, role="gpu")        # training on GPU (fn(config) per item)
```

Each step also picks up that apparatus's `before_each` hooks. The default role is the tick's apparatus (set by `--app` / `--workers` on the CLI). In a notebook, where you already hold apparatus handles, you can instead pass an instance directly: `ctx.run(fn, on=cpu_app)`.

The experiment's `roles=` table binds each label to hardware kwargs, e.g. `roles={"gpu": dict(gpu="L4", timeout=2700, watchdog=120, watchdog_grace=900)}`. Alongside the backend-native knobs, any role (local or Modal) can set `watchdog=` — seconds without *step* progress before the worker aborts itself (`FAILED` with a stack dump) instead of wedging silently until the role `timeout`. Size it past the longest legitimate gap between `emit_progress` step advances; one-off setup before the first emission (tokenization, compilation) is covered by `watchdog_grace=` instead (default: same as `watchdog`), so a slow prep phase doesn't force the watchdog loose. The grace ends at the first emission — emit once real step cadence begins, not before. Leave `watchdog` unset for steps that never emit step progress.

Spans *inside* the loop that legitimately report no steps get the same allowance on demand, via `mini.blocking_phase(label, timeout_s=…)`. `put`, `get` and `get_many` already declare their own, sized from the payload, so a step that checkpoints after its last training step needs nothing — reach for it only when your own code blocks for longer than the step cadence (a long eval, a download you drive yourself). The budget bounds the span rather than exempting it: a span that hangs is still caught, at `timeout_s`.

A role can also set `env=` — environment for the worker, in place *before* the process starts (a Modal container Secret; locally, the task subprocess's env). Reach for it when a library reads its env once at init and a task setting it on itself would be too late: a Modal container is reused across a map's tasks, so whichever task ran first fixes the setting for the rest. `[tool.mini] env` in `pyproject.toml` gives every experiment a baseline (this project uses it for `XLA_FLAGS`, so GPU results reproduce — see [eng/determinism.md](/eng/determinism.md)); a role's `env=` merges over it key by key. It is *not* a credential channel — the values land on the task record; pass tokens through Modal's `secrets=`.

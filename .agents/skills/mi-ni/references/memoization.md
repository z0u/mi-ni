# Memoization: identity, evidence, and the recovery loop

Every `ctx.run`/`ctx.map` call resolves to a durable record that answers two
separate questions:

- **Identity — which task is this?** The *key*: the fn's qualified name plus a
  fingerprint of its inputs. Stable across code edits, so a task's record, logs,
  and results keep one address for the task's whole life.
- **Validity — is the cached result current?** The *evidence* stamped on each
  attempt: a fingerprint of the code the task actually depends on, plus
  `version=`. Stale evidence re-runs the task **in place** — a new attempt on
  the same record, with the old attempt kept in the record's history.

Understanding both is how you keep the "fix a bug, re-run" loop fast and honest.

## How the key and evidence are computed

```
key      = {fn name}-hash(fn's module-qualified name + fingerprint(inputs))
evidence = fingerprint(source(fn) + source(project fns/classes fn calls, transitively)) + version
```

- **Inputs are the identity.** Plain data (dict/list/tuple/str/num, dataclasses,
  pydantic models, enums, `Artifact`s) fingerprints deterministically; a *function*
  passed as data keys by its source, not its object identity. An input with no
  stable encoding (an object whose repr embeds its address) logs a loud warning —
  it can never cache, so the task would relaunch every wake. Renaming a fn is a
  new identity (the old records read `(superseded)`); editing its body is not.
- **Source, not bytes.** Hashing `cloudpickle.dumps(fn)` is tempting (it captures
  by-value dependencies) but its bytes differ across processes — and every agent
  wake is a fresh process, so nothing would ever look current. Both fingerprints
  are deterministic across processes.
- **Evidence is transitive over your own code.** It covers the source of the
  project functions and classes `fn` references — by bare name, as a module
  attribute (`utils.helper()`), from inside a nested lambda/comprehension, or from
  a method of a class the task uses — plus **plain module-level values** the code
  reads (a module-level `LR`, a config table), so editing any of them re-runs the
  task. **Site-packages and the mini framework are excluded**, so library churn
  (or editing mini itself) doesn't bust your cache.
- **`version=` is explicit evidence** — bump it to force a re-run without editing
  code. Like a code edit, the bump lands as a new attempt on the same record.

### What the fingerprint cannot see

Coverage is biased toward over-invalidation (a spurious re-run is visible and
bounded; a stale hit silently poisons results), but some dependencies are
invisible by nature — fold them into the *inputs* instead:

- **Files read at runtime.** Pass an `Artifact` handle (keys by content), not a
  path the task opens.
- **Env vars and machine state.** Pass them as arguments if they affect the result.
- **Attributes on instances** (`self.x` set elsewhere, monkeypatching) and values
  with no stable JSON encoding — not tracked; keep task behavior in code and plain
  data.

### `mini explain`: why did this re-run?

Each attempt stamps its evidence on the record — code hash, input hash, and a
short hash per tracked dependency — and a replaced attempt stays compacted in
the record's history. `mini explain <name> <key>` prints the current evidence
and walks the timeline, naming exactly what moved between attempts:

```
#1 failed     code a1b2c3  !! RuntimeError: divide by zero
#2 done       code d4e5f6  ⇐ helper: changed
```

Use it whenever a memo hit or re-run surprises you.

Why isn't the result keyed on inputs *alone*, with no code tracking? Because
after you fix a bug, pure input-keying would return the _stale, buggy_ result —
the opposite of what the loop needs. Tracking code as validity evidence re-runs
exactly the code that changed, while keeping the task's address (record, logs,
history) stable through the fix.

### Maximise cache hits: pass narrow inputs

The single most effective habit. A task keyed on the entire experiment config
re-runs whenever any unrelated field changes:

```python
ctx.map(train, [(whole_config,) for ...])   # re-runs on ANY config change
ctx.map(train, [(lr, vocab_size) for ...])  # re-runs only when lr / vocab_size change
```

Keep `main` cheap and deterministic (it re-runs every wake), and fold RNG seeds
into a task's inputs so the same inputs really do produce the same result.

## Fix / prune / retry

<!-- prettier-ignore -->
| You want to… | Do this | What re-runs |
| --- | --- | --- |
| Fix a bug in a step | Edit the fn, `mini run` | Every stale cell of that fn — in place, same keys (FAILED cells relaunch automatically); *other* steps stay hits |
| Fix a bug without redoing finished cells | Edit the fn, `mini run --keep-stale-done` | Only cells that never finished; DONE results are kept and badged `(stale code — kept)` |
| Add a config to a sweep | Append to `configs`, `mini run` | Only the new key |
| Remove a config | Delete it from `configs` | Nothing — its old record shows `(superseded)` |
| Re-run a finished step | Edit the fn, or pass `version=` | That step — a new attempt on the same record |
| Recover a failed step | `mini logs`, fix, `mini run` — or `mini retry` if the failure was flaky (no code change) | The stale (or reset) FAILED/CANCELLED tasks |

### Hotfix a sweep in bounded time

Mid-sweep, a bug fails 20 of 100 cells while 80 finish fine. Because keys are
identity, fixing the fn doesn't orphan anything — every cell keeps its key, and
the tick judges each record against the new evidence:

- **Default** (`mini run`): all 100 cells are stale, so all 100 re-run. Honest,
  but it re-pays for the 80 good results.
- **Bounded** (`mini run --keep-stale-done`): the 80 DONE cells are served as-is
  (their results predate the fix — `status` badges them `(stale code — kept)`,
  and the tick records them in the run's meta), and only the 20 failed cells
  re-run with the fixed code. No `retry` needed: a FAILED record whose code has
  since changed relaunches automatically — the fix is what it was waiting for.

Keeping stale DONE results is a *judgment call* — it asserts the edit didn't
change what the finished cells computed. The default deliberately re-runs them
(bias to over-invalidate); reach for the flag when you know the fix only matters
to the cells that failed.

### Superseded records

Renaming a task fn or removing a config changes what the DAG *requests*, leaving
old records behind under keys no wake will ask for again. Each tick persists the
set of keys the DAG requested; the read commands aggregate over that set, showing
the orphans as `(superseded)` without letting them poison the run's state — a
completed run reads DONE even if an old key once settled FAILED. `retry` skips
superseded records too (resetting one would plant a phantom that never runs);
target one explicitly with `--key` if you really mean it. (Editing a fn's *body*
no longer supersedes anything — the re-run lands on the same record.)

### Failure is terminal by design

`FAILED` and `CANCELLED` are terminal *under the code that produced them*: a
plain `mini run` will **not** relaunch them. This is deliberate — a
deterministic failure shouldn't busy-loop, and a fix should be intentional.
Recovery takes one of:

- fix the code and `mini run` — the record's evidence is stale, so it relaunches;
- bump `version=` — same effect, without an edit;
- `mini retry <name>` — for a *flaky* failure (nothing changed): resets all
  FAILED/CANCELLED tasks (`--key <key>` for one), then advances the DAG.

The traceback lives on the I/O plane (`mini logs <name> <key>`); the record
carries the last error line for a quick scan in `status`, and each healed
record keeps its failed attempts in history (`mini explain`).

### A failed item fails its `map` — unless you allow partials

By default `ctx.map` raises `Pending` until _every_ item has settled. Once the
fan-out settles, any item that settled `FAILED`/`CANCELLED` makes the map raise an
**`ExceptionGroup` of `TaskFailed`** — all of them at once, so you see every
failure, not just the first. (`ctx.run`, being a single step, raises a bare
`TaskFailed`.) The group carries each worker's stored traceback; handle it with
`except* TaskFailed`. That's the right default when every cell matters — and since
a settled failure won't relaunch, raising is how the DAG gives up instead of
spinning. Recover with `retry`.

When some failures are expected — a bad hyperparameter region, an OOM at the
extreme, a preempted container — pass **`allow_partial=True`**. The map still
waits for in-flight tasks to settle, but then returns instead of raising on the
failures. The result list stays index-aligned with the inputs, with the `MISSING`
sentinel in each failed/cancelled position:

```python
from mini import MISSING

results = ctx.map(train, configs, allow_partial=True)   # [r0, MISSING, r2, ...]
ok = [(c, r) for c, r in zip(configs, results, strict=True) if r is not MISSING]
best = min(ok, key=lambda cr: cr[1]['val_loss'])
```

`MISSING` is a falsey singleton distinct from `None` (which a task may legitimately
return), so `r is MISSING` and `[r for r in results if r]` both work. The failed
cells are still terminal — `retry` reruns them, and the next wake fills their real
results — `allow_partial` just unblocks the map's downstream in the meantime.

## Reading results without re-running

A report or a status check must not `tick` (that launches work). Read the durable
store directly via the apparatus:

```python
from mini import LocalApparatus, RunState

store = LocalApparatus('my-exp').memo_store()    # ModalApparatus(...).memo_store() for --app modal
records = store.records()                          # per-task state/metrics
done = [store.result(r['key']) for r in records if r.get('state') == RunState.DONE]
```

This is exactly what `mini status`/`results` and a `report.py` notebook do.

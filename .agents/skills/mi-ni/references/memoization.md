# Memoization: keys, caching, and the recovery loop

Every `ctx.run`/`ctx.map` call is content-addressed. The key decides whether a
call is a cache hit, so understanding how it's computed is how you keep the
"fix a bug, re-run" loop fast and honest.

## How the key is computed

```
key = fingerprint(source(fn) + source(project fns/classes fn calls, transitively)) + fingerprint(inputs)
```

- **Source, not bytes.** Hashing `cloudpickle.dumps(fn)` is tempting (it captures
  by-value dependencies) but its bytes differ across processes — and every agent
  wake is a fresh process, so it would miss the cache _every wake_. The source
  fingerprint is deterministic across processes.
- **Transitive over your own code.** It includes the source of the project
  functions and classes `fn` references, so editing a helper your task calls
  invalidates the task. **Site-packages and the mini framework are excluded**, so
  library churn (or editing mini itself) doesn't bust your cache.
- **Inputs are part of the key.** `pickle`-stable inputs (dict/list/tuple/str/num)
  fingerprint deterministically.
- **`version=` is an explicit override** added to the hash — bump it to force a
  re-run without editing code.

Why not key on inputs alone? Because after you fix a bug, pure input-keying
returns the _stale, buggy_ result — the opposite of what the loop needs. Source
fingerprinting re-runs exactly the code that changed.

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
| Fix a bug in a step | Edit the fn, `mini run` | That step (new source key); siblings stay hits |
| Add a config to a sweep | Append to `configs`, `mini run` | Only the new key |
| Remove a config | Delete it from `configs` | Nothing — it's just not requested |
| Re-run a finished step | Edit the fn, or pass `version=` | That step |
| Recover a failed step | `mini logs`, fix, `mini retry` | The reset (FAILED/CANCELLED) tasks |

### Failure is terminal by design

`FAILED` and `CANCELLED` are terminal: a plain `mini run` will **not** relaunch
them. This is deliberate — a deterministic failure shouldn't busy-loop, and a
fix should be intentional. Recovery takes one of:

- `mini retry <name>` — resets all FAILED/CANCELLED tasks (`--key <key>` for one),
  then advances the DAG;
- bump `version=` or edit the fn — a new key, so `run` launches it fresh.

The traceback lives on the I/O plane (`mini logs <name> <key>`); the record
carries the last error line for a quick scan in `status`.

### A failed item blocks its `map` — unless you allow partials

By default `ctx.map` raises `Pending` until _every_ item is `DONE`. A settled
`FAILED` item no longer relaunches (good) but still blocks the map from returning,
so the steps after it can't proceed until you `retry` or prune it. That's the
right default when every cell matters.

When some failures are expected — a bad hyperparameter region, an OOM at the
extreme, a preempted container — pass **`allow_partial=True`**. The map still
waits for in-flight tasks to settle, but then returns instead of blocking on the
failures. The result list stays index-aligned with the inputs, with the `MISSING`
sentinel in each failed/cancelled position:

```python
from mini import MISSING

results = ctx.map(train, configs, allow_partial=True)   # [r0, MISSING, r2, ...]
ok = [(c, r) for c, r in zip(configs, results) if r is not MISSING]
best = min((r for _, r in ok), key=lambda r: r['val_loss'])
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

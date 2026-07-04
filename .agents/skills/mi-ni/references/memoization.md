# Memoization: identity and evidence

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
The loop's mechanics — the fix/prune/retry table, partial failures, reading
results — are in [recovery.md](./recovery.md).

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
ctx.map(train, whole_configs)      # re-runs on ANY config change
ctx.map(train, lrs, vocab_sizes)   # re-runs only when lr / vocab_size change
```

(`ctx.map` zips its iterables Executor-style — `train(lr, vocab_size)` per pair,
mismatched lengths raise. A single iterable passes each element as one argument,
tuples included.)

Keep `main` cheap and deterministic (it re-runs every wake), and fold RNG seeds
into a task's inputs so the same inputs really do produce the same result.

## Recovering, fixing, retrying

Once you understand identity and evidence, the operational loop — fixing a bug and
re-running in bounded time, pruning superseded records, recovering terminal
failures, handling partial `map` failures, and reading results without re-running
— is in [recovery.md](./recovery.md).

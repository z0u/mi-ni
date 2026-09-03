# Memoization: identity and evidence

Every `ctx.run`/`ctx.map` call resolves to a durable record that answers two separate questions:

- Identity, which task is this? The *key*: the fn's qualified name plus a fingerprint of its inputs. Stable across code edits, so a task's record, logs, and results keep one address for the task's whole life.
- Validity, is the cached result current? The *evidence* stamped on each attempt: a fingerprint of the code the task actually depends on, plus `version=`. Stale evidence re-runs the task in place: a new attempt on the same record, with the old attempt kept in the record's history.

Understanding both is how you keep the "fix a bug, re-run" loop fast and correct. The habits that follow from it — narrow inputs, a cheap `main`, folded RNG seeds, where to put a new helper — are in [authoring.md](./authoring.md#write-cache-friendly-experiments); the mechanics of the loop (the fix/prune/retry table, partial failures, reading results) are in [recovery.md](./recovery.md).

## How the key and evidence are computed

```
key      = {fn name}-hash(fn's module-qualified name + fingerprint(inputs))
evidence = fingerprint(source(fn)
                       + source(project fns/classes fn calls, transitively)
                       + source(what fn imports in its own body)
                       + source(the __init__.py of each package on the way there)
                       + source(the import-time statements of each module reached)) + version
```

`joblib.Memory` and friends stop at the first line of that — why this is `mini`'s own code rather than a library's is recorded in [eng/decisions.md](/eng/decisions.md).

- Inputs are the identity. Plain data (dict/list/tuple/str/num, dataclasses, pydantic models, enums, `Artifact`s) fingerprints deterministically; a *function* passed as data keys by its source, not its object identity. An input with no stable encoding (an object whose repr embeds its address) logs a loud warning: it can never cache, so the task would relaunch every wake. Renaming the task fn — or moving it to another module, since the key hashes its module-qualified name — is a new identity, and the old records read `(superseded)`; editing its body is not.
- Source, not bytes. Hashing `cloudpickle.dumps(fn)` is tempting (it captures by-value dependencies) but its bytes differ across processes, and every agent wake is a fresh process, so nothing would ever look current. Both fingerprints are deterministic across processes.
- Evidence is transitive over your own code. It covers the source of the project functions and classes `fn` references — by bare name, as a module attribute (`utils.helper()`), from inside a nested lambda/comprehension, or from a method of a class the task uses — plus plain module-level values the code reads (a module-level `LR`, a config table), so editing any of them re-runs the task. Site-packages and the mini framework are excluded, so library churn (or editing mini itself) doesn't bust your cache.
- Deferred imports count too, at the same granularity. A task that imports inside its own body — the usual way to keep the driver and CLI light when the import pulls jax — gets the source of each *name* it imports, plus whatever those names reference, transitively. `from experiment.compute.data_pipelines import save_data` tracks `save_data` and its callees, not the other twenty functions in the file; `from experiment.data import preparation as prep` followed by `prep.tokenize_data(...)` tracks `tokenize_data`. Modules are located by searching `sys.path` and read as text, never imported, so the deferred import stays deferred. Where source can't say what a name binds — a star-import, a name defined inside an `if`, an alias passed around as a value rather than dotted into — the whole module counts instead. Same for a plain `import x`: the name reached through it isn't readable off the statement. So `from x import y` is both cheaper and more precise than `import x`.
- `version=` is explicit evidence: bump it to force a re-run without editing code. Like a code edit, the bump lands as a new attempt on the same record.

### Granularity: a definition, not a file

The unit of evidence is the definition a task reaches, so the rest of the module holding it stays invisible. **Adding a function to an existing module leaves the cache alone** for every task that doesn't call the new function: the manifest holds `helpers:helper`, and the neighbour never enters it. So there's no cache argument for starting a new module rather than extending one — and the habit costs re-runs, because the deferred walk keys each dependency by `module:symbol`: move a helper to a new file and everything importing it re-runs, with the source byte-identical.

Three places where a whole file is the unit, so adding to them does re-run what's downstream:

- **A package `__init__.py`**, for every import beneath it. `from experiment.compute.data_pipelines import save_data` folds in `experiment/__init__.py` and `experiment/compute/__init__.py` whole, because they execute on the way down and can change what the task computes (ours sets `XLA_FLAGS`). Worth keeping thin. Where the import is written makes no difference: a deferred import reads the chain off the dotted name, a module-scope one reads it off the helper's `__module__`, and both also fold in the defining module's own import-time statements.
- **A class**, methods the task never calls included: a class is fingerprinted as one block of source.
- **A module reached without a readable name** — plain `import x` in a task body, a star-import, an alias used bare rather than dotted into. Deliberate, per the deferred-imports bullet above.

### What the fingerprint cannot see

Coverage is biased toward over-invalidation (a spurious re-run is visible and bounded; a stale hit silently poisons results), but some dependencies are invisible by nature — fold them into the *inputs* instead:

- Files read at runtime. Pass an `Artifact` handle (keys by content), not a path the task opens.
- Env vars and machine state. Pass them as arguments if they affect the result. The exception this project makes deliberately is `XLA_FLAGS`, which decides whether a GPU reduction is deterministic: it's set project-wide via `[tool.mini] env` and recorded on every attempt (`env.numerics_env`) rather than folded into the key, so turning determinism on didn't invalidate four published sweeps. The reasoning, and the measurements behind it, are in [eng/determinism.md](/eng/determinism.md).
- Attributes on instances (`self.x` set elsewhere, monkeypatching) and values with no stable JSON encoding — not tracked; keep task behavior in code and plain data.
- Modules the driver process can't find. The `sys.path` search reads "no source file" the same way for a C extension and for project code that isn't installed, and the second kind then contributes nothing to the evidence — so the task looks dependency-free and caches forever. It can't be resolved from inside the walk (the source genuinely isn't there to read), so it warns instead: `no source found for 'x' on sys.path, and it is neither stdlib nor an installed package`. On your own code that means the driver's environment is missing it — check the editable install or `PYTHONPATH` — and until it's fixed, edits to that module will not re-run the task. Two shapes reach the same warning while being fine as they are, and both are worth recognizing so the message doesn't send you after an install that was never broken: an optional dependency behind `try/except ImportError` warns whenever it's absent, which is the case that code handles; and an `if TYPE_CHECKING:` import of a package that isn't installed at runtime can't affect what the task computes. Telling either apart from a real hole needs the source context that says the import is guarded, and the walk reaches a task body as bytecode, where the `try` and the `if` are already jumps. A PEP 420 namespace package used to warn too, and no longer does — a directory on `sys.path` with no `__init__.py` has no source of its own, while its submodules resolve and join the evidence normally.

### `mini explain`: why did this re-run?

Each attempt stamps its evidence on the record — code hash, input hash, and a short hash per tracked dependency — and a replaced attempt stays compacted in the record's history. `mini explain <name> <key>` prints the current evidence and walks the timeline, naming what moved between attempts:

```
#1 failed     code a1b2c3  !! RuntimeError: divide by zero
#2 done       code d4e5f6  ⇐ helper: changed
```

Use it whenever a memo hit or re-run surprises you.

Why isn't the result keyed on inputs *alone*, with no code tracking? Because after you fix a bug, pure input-keying would return the _stale, buggy_ result — the opposite of what the loop needs. Tracking code as validity evidence re-runs exactly the code that changed, while keeping the task's address (record, logs, history) stable through the fix.

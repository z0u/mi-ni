# Reproducible GPU runs

*Part of the [engineering notes](./README.md).*

Two tasks with the same code and the same inputs should produce the same result. On a GPU they don't, by default — and because the memo keys a task on the *content* of its inputs, that turns one re-run upstream into a re-run of everything downstream.

## What we measured

150 training steps of the real `train_step` (nGPT, d64-L4, batch 64×128, one seed), three replicas in three separate `single_use_containers=True` L4 containers, `us-east`. The probe hashes the final weights; the flags ride in the container environment.

| `XLA_FLAGS` (beyond the CPU scheduler flag) | distinct digests | seconds |
| --- | --- | --- |
| *(none)* | **3 of 3** | 29.0, 29.8, 29.0 |
| `deterministic_ops` | 1 | 49.6, 79.3, 49.0 |
| `deterministic_ops` + `autotune_level=0` | 1 | 49.3, 83.7, 50.2 |
| all three | 1 | 49.1, 85.3, 48.3 |
| **`deterministic_ops` + `exclude_nondeterministic_ops`** (shipped) | 1 | 82.9, 48.0, 49.0 |

The baseline's three final losses were 4.147422, 4.147426, 4.147430 — a drift in the sixth significant figure, which is invisible in a plot and total for a hash. All four deterministic settings landed on the *same* digest (`6d18d792…`), so there's one canonical answer they agree on rather than four flavours of stable.

One replica per batch runs 1.6–1.7× slower than its siblings, in the deterministic rows only. It was the middle replica three times running and then the first one, so it reads as an unlucky container rather than anything about the flags — but it does mean a run's slowest cell isn't well predicted by its fastest. The cost figure to plan around is the fast replicas: **~1.7×** (29s → 48s) on a cell this size.

## What we set, and why only that

`[tool.mini] env` in `pyproject.toml` carries the CPU scheduler flag plus:

- **`--xla_gpu_deterministic_ops=true`** — the one that does the work. It makes scatter deterministic through XLA's scatter-determinism expander (an optimized rewrite, so it's much cheaper than the fallback). Our model has one big scatter: the token-embedding gradient. Without the flag those accumulate in whatever order the atomics land, which is a different order every run.
- **`--xla_gpu_exclude_nondeterministic_ops=true`** — the broader guard, for ops the scatter expander doesn't cover. It measured free at our sizes, and it's what keeps a future model that reaches for some other atomically-reducing op from quietly reintroducing drift.

The pair was measured on its own (last row) rather than inferred from the three-flag run, so what's configured is what was tested.

**Autotuning stays on.** It's the other classic source of GPU nondeterminism — the algorithm pick comes from measured timings, so a noisy host can pick differently — but turning it off changed neither the digest nor the time here. It has a plausible cost on larger models (a worse GEMM), so we're not paying it for a benefit this project hasn't observed. If a future run drifts *with* the flags above, `--xla_gpu_autotune_level=0` is the next thing to reach for.

Not established here: whether the digest holds across GPU *models*. Determinism is a per-device-class property, and the role tables pin `gpu="L4"` anyway.

## Why the environment, and not a line of Python

A Modal container is reused across the tasks of a map, and XLA parses `XLA_FLAGS` once, when its backend comes up. So a task that sets the variable on itself only wins if it's the first task on that container — every later one silently inherits whatever the first one had. (This is why a package `__init__`'s scheduler flag was never reliable on Modal; it's still right for the interactive path, where a notebook is one process and imports the project package before touching jax.)

So it's an apparatus option: `env=`, which `ModalApparatus` turns into a container Secret and `LocalApparatus` overlays onto the task worker's subprocess. `[tool.mini] env` supplies it project-wide — the same shape as `region` — and a role's own `env=` merges over it key by key. It is deliberately **not** a credential channel: the values are recorded on the task record, and `secrets=` is still there for tokens.

## Why this isn't in the memo fingerprint

Flipping the flags changes results without changing code or inputs, so the memo can't see it. We chose not to make it evidence:

- Turning determinism on today would invalidate every DONE record in the project and re-run four published sweeps to reproduce numbers we already have. The existing records are honest about what they are; they just came from the older regime.
- The question this actually needs to answer is "what was *this* number computed under?", which is provenance, not identity. `compute_env` records `XLA_FLAGS` on every attempt (`env.numerics_env`, alongside the GPU model and region), read from the worker's own environment — so a setting that never reached the container reads as absent instead of as what the client meant to send.

If we ever do want a re-run on a flag change, `version=` on the task is the explicit lever, and it lands as a new attempt on the same record.

## The JAX version moves the digest too

The flags are not the only thing outside the fingerprint that changes the number. Measured while upgrading JAX for #73 — same code, same inputs, same seed, same L4, same `XLA_FLAGS`, 150 steps of the real `train_step` on nGPT d64-L4, three single-use containers per version:

| jax / jaxlib | digest | final loss |
| --- | --- | --- |
| 0.10.1 | `e4b7c106…` | 5.531451225280762 |
| 0.11.0 | `884ee00c…` | 5.531452655792236 |

One distinct digest within each version, so 0.11 is every bit as reproducible as 0.10 — it just lands somewhere else, about a part in 4×10⁶ away. That is the same magnitude as the nondeterminism the flags were introduced to remove, which is worth sitting with: the flags buy reproducibility *within* a pinned environment, not across one.

The memo cannot see this. `_is_project_file` excludes site-packages, so no library source reaches the manifest, and `task_key_parts` composes identity from the fn's module-qualified name and an input fingerprint — neither of which mentions a package version. Checked directly: no `jax`, `jaxlib` or version string appears anywhere in a task's key or its evidence. So the failure mode is not a spuriously invalidated key (harmless — just a re-run) but the quiet one: **the same key mapping to a numerically different result after an upgrade.** A DONE record written under 0.10.1 keeps serving its old result, and a task re-run for any other reason silently produces the new one, under the key that already meant the old.

The same reasoning as above still applies — this is provenance, not identity, and `compute_env` is where it belongs. But unlike `XLA_FLAGS`, a version bump arrives with an ordinary dependency upgrade rather than a deliberate flag change, so it wants a deliberate gate. When a numerics-relevant package moves, either accept that records straddle two regimes and say so where the numbers are published, or bump `version=` on the affected tasks to force a clean re-run.

## The gate: detection, not invalidation

`compute_env` records the version of each package in `mini.runs._NUMERICS_PACKAGES` (`jax`, `jaxlib`, `numpy`) on every attempt, as `env.numerics_packages`. That is what makes the straddle *visible*: `numerics_drift` compares a record's versions against the ones installed now, and two readers act on it.

- A **tick** notes every memo hit whose result predates an upgrade, and logs one warning naming what moved (`jax 0.10.1 → 0.11.0`) and how many hits it covers. Said once per answer, per process — a watching driver ticks every few seconds and the answer doesn't change — but a later wake that reaches more drifted hits (a suspended tick only walks the DAG to its suspension point) speaks again with the larger count. A run that straddles an upgrade can carry several recorded versions of one package; the warning names them all (`jax 0.9.0/0.10.1 → 0.11.0`).
- **`mini status`** re-derives the same comparison from the records alone (`⚠ N result(s) computed under different numerics: …`, and a `numerics_drift` object under `--json`). No import, no tick, no fingerprinting, so it still answers weeks later, which is when someone is usually asking where a number came from.

Nothing re-runs on its own, which is the point: the levers stay `version=` and an honest note beside the published numbers. Two things this does not see. A package outside the list moves silently — the list is what a measurement has implicated, and adding a name only affects records written afterwards. And the comparison is against the *driver's* installed set, not the container's; the Modal image is frozen from the same lock, so they agree unless something has been pinned apart deliberately. Records written before this shipped carry no versions and read as quiet rather than as drifted: no evidence, rather than evidence of no change.

## The alternative we didn't take: tolerating small differences

The tempting move is to leave the GPU alone and teach the DAG that 4.147422 and 4.147430 are the same number. It doesn't work, for a reason worth writing down: near-equality isn't an equivalence relation. `a ≈ b` and `b ≈ c` doesn't give `a ≈ c`, so there's no way to bucket values into hashable classes. Concretely:

- **Rounding before hashing** just relocates the problem to the bucket boundaries. Two values a hair apart that straddle one still hash differently, so misses become rarer *and* less predictable — the worst combination for a cache you're trying to trust. It also does nothing for the input that dominates here, which is a checkpoint blob.
- **Keeping the previous result when the new one is within tolerance** does hold the downstream keys still, and it's the only version that's mechanically compatible with content addressing. But it serves a result the current code didn't produce, it needs a tolerance per result *shape* (there isn't a meaningful one for checkpoint bytes), and it makes the DAG's answer depend on run history rather than on inputs — which is the property that makes a memo worth trusting.
- **Keying downstream on the upstream's identity rather than its bytes** — pass the config and seed, stash the weights beside it — is the honest version, and it's the fallback if determinism ever costs too much. It's also a deliberate trade: a genuine change in the upstream result stops invalidating downstream. Worth pairing with recording the checkpoint digest, so a surprise is at least detectable after the fact.

Determinism is simply the cheaper lever: one flag, one place, ~1.7× on the training role, and the DAG's equality test keeps meaning what it says.

## Re-measuring

The probe isn't checked in — it needs a GPU, so it can't be a test. It's ~70 lines: map the real `train_step` over N replicas with `single_use_containers=True` and `.w(env={"XLA_FLAGS": ...})`, hash `jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))`, and compare digests across containers. Worth re-running when the model grows a new kind of op, or when the 1.7× starts to hurt.

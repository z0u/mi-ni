# Recovery: fix, prune, retry

The fix-a-bug-and-re-run loop, and how a task's identity keeps it fast and honest.
This is the mechanics; the identity/evidence *model* it rests on is in
[memoization.md](./memoization.md), and the operational safety of editing under live
workers (cost, blast radius) is in [running.md](./running.md#hotfix-safety-avoid-double-spending).

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

Superseded records linger until reclaimed: `mini gc <name>` prints a sweep plan
(superseded records with their result dirs, files from replaced attempts, stale
staged calls) and `--apply` deletes it. It never touches a current record — a
DONE result is a future memo hit, and deleting a FAILED one would silently turn
a terminal failure into a relaunch — and it collects superseded records only
once the last tick ran the DAG to completion with nothing left in flight.
Local backend only for now (Modal Dict entries self-expire; see #15).

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
best = min(ok, key=lambda cr: cr[1]["val_loss"])
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

store = LocalApparatus("my-exp").memo_store()    # ModalApparatus(...).memo_store() for --app modal
records = store.records()                          # per-task state/metrics
done = [store.result(r["key"]) for r in records if r.get("state") == RunState.DONE]
```

This is exactly what `mini status`/`results` and a `report.py` notebook do.

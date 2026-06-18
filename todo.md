# TODO

Deferred design/work items. See [the proposal](./docs/proposals/automated-research.md)
and [notes/agentic-experiments.md](./notes/agentic-experiments.md) for context.

## Memoized orchestration

- **`allow_partial=` for `ctx.map`.** Today `map` raises `Pending` if *any* item
  isn't `DONE`, and a deterministically-failing item relaunches every tick → it
  blocks the whole map forever. We want a way to proceed on partial results.
  Subtlety to design around: downstream code often `zip`s configs with results,
  so we can't just drop missing items (it'd misalign). Options: return positional
  sentinels (`None`/a `Missing` marker) for un-`DONE` items, or return a
  config-keyed mapping instead of a positional list. Decide and document
  (skill + README + test).

- **Document the fix/prune/retry semantics** (skill + README + test):
  - remove an item from configs → not requested; unchanged items are memo hits.
  - change an item (fold the seed into args) → new key → only that item re-runs.
  - a `FAILED` item auto-relaunches on the next tick (only that key).
  - force a re-run of a `DONE` item via `version=` or by editing the fn.

## Polling / monitoring

- **Cheap polling for large sweeps.** Cache settled (`DONE`/`FAILED`/`CANCELLED`)
  records client-side — they're immutable — and poll only the unsettled subset.
  Keep control-plane records small (latest scalars; history to the I/O plane).
  With per-job keys, read only active jobs; consider a last-writer-wins run-summary
  key for cheap top-level polling. Relevant once the control plane is a `modal.Dict`.

- **Interactive monitor to replace `LocalQueue`.** A long-running `mini watch <exp>`
  that loops on the durable read path (`MemoStore.records()` / control plane) and
  renders Rich until settled — no queue, works against runs it didn't launch.

## Compute decoupling (apparatus)

**Done (local-only slice).** Compute is injected at execution, not stored on the
experiment:
- `Experiment` no longer has an `apparatus` field (nor `make_apparatus`,
  `data_dir`, `before_hooks`); `Experiment.submit(apparatus)` and
  `tick(exp, apparatus)` take it explicitly. CLI `--app`/`--workers` build it via
  `_build_apparatus` (`--app local` only; `modal` raises "not supported yet").
- Per-step override plumbed: `ctx.run(fn, *args, on=apparatus)` /
  `ctx.map(fn, items, on=apparatus)`, defaulting to the tick's apparatus; per-step
  `before_each` hooks come from whichever apparatus runs the step. Covered by
  `test_per_step_apparatus_uses_its_hooks`.
- Decided **against a generic `Requirements` type** (over-engineered: it would
  need a translation layer to each backend's native options — Modal's
  `function()` alone has gpu/cpu/memory/timeout/retries/region/…). The apparatus
  owns its native knobs; selection happens at the edge (CLI flag / notebook code).

**Still to do:**
- **Wire compute through the memoized path.** `MemoStore.launch` still hardcodes a
  local `subprocess.Popen` and never consults the apparatus, so `on=` currently
  only differentiates *hooks* locally — not *where* work runs. The Modal backend
  (and meaningful per-step `on=`) needs `launch` to spawn via the apparatus.
- **Role labels for file-based experiments.** `on=apparatus` only works from a
  notebook (which holds apparatus handles); a file experiment loaded by the CLI
  has none. Plan: abstract *role* labels (`ctx.map(..., role='gpu')`) that the
  driver/CLI maps to concrete apparatuses, so `main` stays backend-agnostic.
- **Capture compute-environment metadata in the run/task records** (what it
  actually ran on) — separate from backend selection above.

## CLI / ergonomics

- **Expose a real entrypoint** — `mini launch …` / `./go mini …` rather than
  `python -m mini …` (notes/agentic-experiments.md, "CLI" section). See the
  `bin/` wrapper convention.

## Cross-experiment composition

- **Consume experiment A's output from experiment B.** Within one experiment the
  memoized DAG already handles `f → g`. For genuinely separate experiments, the
  clean path is value-level: import A's experiment and `tick` it (instant memo
  hit if done) to get its return value. Sugar like `get_data_dir(other)` for
  reaching another experiment's volume is the open question — lean toward
  explicit shared locations over reaching into another experiment's private dir.

## Modal backend

- `ModalControlPlane` backed by a named `modal.Dict` behind the `ControlPlane`
  ABC; split `ControlPlane`/`LocalControlPlane`/`ModalControlPlane` into their own
  `control.py` when the Modal class lands (move both at once).
- Detached execution: `spawn_map` under `app.run(detach=True)`; persist the
  `FunctionCall` id; poll per index.

## Housekeeping (from the proposal's "Deferred / open")

- Auto-teardown on a wall-clock budget so a forgotten detached run can't burn
  money indefinitely.
- Garbage-collect old run records / named Dicts / Volume run dirs.

# Project-scoped storage for experiments

**Status:** proposed (2026-06-20). Captures the design agreed while adding
`HFStore`; informs a follow-up that touches the apparatus + CLI.

## Problem

We have two storage tiers: compute-attached volumes (`Volume`, per experiment)
and published artifacts (Git LFS → Pages, now also `HFStore` → HF Hub). What's
missing is a way for **experiment B to reuse experiment A's outputs** — e.g.
`gpt.py`, `gpt_sweep.py`, and `gpt-sweep/experiment.py` all prepare the *same*
tokenized Pride and Prejudice corpus, three times over.

The near-miss: the memo key is already content-addressed and
**experiment-name-independent** — `fingerprint(fn, args, version)` =
`code-fingerprint(fn + transitive project sources)` + `input-fingerprint(args)`.
So identical prep in A and B produces the *same key*. What keeps them apart is
that the **store and volume are scoped by experiment name**:

- Local: `.mini/<name>/…`
- Modal: volume `<name>` + `modal.Dict` `mini-cp-<name>`

Identical keys land in different stores → B recomputes what A already did.

Two things compound it:

- **Routing doesn't move storage.** `on=`/`role=` vary *hardware* (`base.w(...)`,
  same name → same volume/dict); `Ctx` always reads/writes the tick's store
  (`self.store`). So `role=` can't be repurposed for sharing. (Aside: a
  cross-name `on=ModalApparatus('other')` is subtly broken today — the worker
  writes to `other`'s volume/dict while `Ctx` reads the tick store, so the
  result is never seen.)
- **Artifacts live in the volume, not the memo result.** `prepare_data` writes
  the corpus into `get_data_dir()` and returns only metadata. So sharing prep
  means sharing the **volume bytes**, not just the record.

## Decision

**One volume + one memo store per _project_, not per experiment.** "Experiment"
stops being a storage boundary and becomes an orchestration/identity boundary: a
named `main(ctx)` DAG plus a tag on records.

Why this over a dedicated shared-scope primitive + read-only input mounts:

- The memo is already content-addressed, so a shared store gives
  **cross-experiment compute dedup for free** — no `scope=`/`materialize`
  plumbing required to get a hit.
- A shared volume makes the bytes readable everywhere via `get_data_dir()` — the
  read-only input-mount machinery evaporates.
- It matches `data_root()`, which already anchors `.mini` at the *project* root.

### On losing read-only isolation

Read-only mounts protect *mutable shared paths*. The cleaner substitute is to not
have them: **content-address shared artifacts**, the way the memo already does
for results under `_memo/<key>/`. The only real hazard with one volume is two
experiments writing *different* data to the *same fixed path* (e.g.
`processed/data.bin`) and clobbering. Writing datasets under a key-derived path
(`datasets/<key>/…`) makes distinct datasets diverge and identical ones coincide
— sharing *and* collision-safety, no enforcement needed. (Today everything uses
Pride and Prejudice, so the clobber is latent — worth designing out now.)

## Project identity

Derive the project id from the **same directory `data_root()` resolves to** (the
nearest `pyproject.toml`/`.git` walking up from cwd), so local `.mini` and the
Modal namespace always agree:

- id = that `pyproject.toml`'s `[project].name`; fall back to the anchor dir name.
- `MINI_PROJECT` env var overrides.

**Monorepos:** a uv workspace member has its own `pyproject.toml`, so the nearest
marker is the member and its `[project].name` is the member name — each member
gets its own storage namespace automatically, consistent with `data_root()`
anchoring at the nearest marker. Set `MINI_PROJECT` at the repo root to force a
single shared namespace across members.

Naming: local store at `.mini/store`; Modal volume `mini-<project>` and dict
`mini-cp-<project>`. The Modal *app* name can stay per-experiment for dashboard
grouping — only the volume/dict go project-wide.

## Required changes

1. **Tag records with their experiment.** `mark_running` already writes `fn`; add
   `experiment=<name>`.
2. **CLI filters by tag.** `ls` groups by it; `status`/`watch` filter by it; and
   importantly `cancel`/`retry` must scope to it — `cancel` currently settles
   *every* record in the store, which would now span all experiments.
3. **Reports filter `records()`** by experiment (else a report sees every
   experiment's tasks).
4. **Decouple name → resource** in `LocalApparatus`/`ModalApparatus`: volume/dict
   come from the project id, not the experiment name.
5. **Content-addressed dataset path helper** (the read-only replacement above).
6. **`materialize(fn, *args, on=app)`** as the notebook front-door: run-or-reuse
   against the project store, return the result. Trivial once storage is shared.

## HF-from-volume (really large assets)

`HFStore.publish` is client-side today: volume → laptop → HF (a double hop that
may not fit on an agent's disk). No core change needed — publishing is just a
step you run on the apparatus, with the volume mounted, so bytes go volume → HF
server-side:

```py
def publish_model():
    return HFStore('z0u/mi-ni-artifacts').publish(get_data_dir() / 'model', 'nanogpt/model')

app.w(secrets=[modal.Secret.from_name('huggingface')]).run(publish_model)
```

Needs `HF_TOKEN` delivered as a Modal secret via `.w(secrets=...)`; add a thin
`publish_step` helper + a worked example.

## Rollout

- [x] Extract `experiment/corpus.py`; point `gpt.py`, `gpt_sweep.py`, and
      `gpt-sweep/experiment.py` at the one symbol. (Correct under any model.)
- [ ] Record tagging (`experiment=`) + CLI filtering (`ls`/`status`/`watch`/
      `cancel`/`retry`) + report scoping.
- [ ] Project-id resolution + name→resource decoupling in both apparatuses.
- [ ] Content-addressed dataset path helper.
- [ ] `materialize()` notebook front-door.
- [ ] HF remote `publish_step` + example + Modal-secret docs.
- [ ] Re-run `gpt.py` / `gpt_sweep.py` to refresh their published HTML (needs
      GPU/Modal; the corpus extraction left the committed HTML stale vs source).

## Open questions

- Concurrent experiments sharing one Modal volume: ever want hard isolation?
  (`MINI_PROJECT` is the escape hatch.)
- Store GC: delete-by-tag per experiment vs. whole-store reset.

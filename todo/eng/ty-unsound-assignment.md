---
status: done
tags: [tooling, typing]
opened: 2026-09-15
closed: 2026-09-15
---
# Consider enabling ty's `unsound-assignment` rule

ty has an opt-in rule (`unsound-assignment`, off by default) that flags assignments where the value's inferred type is *assignable* to the declared type but not a *subtype* of it. In practice that means one thing: an `Any` landed in a typed slot. `Any` is assignable to everything, so the normal `invalid-assignment` check waves it through, and a wrong type can then travel a long way before anything notices.

A trial run (`uv run ty check --warn unsound-assignment`) finds 28 across the repo, spread over 19 files — a few in `src/mini/`, a few in `src/sca/`, a couple in `scripts/` and two reports.

The catch is that most of them are `Any` arriving from an untyped third-party call rather than a mistake of ours. `eqx.apply_updates` returns `Any`, so `model = eqx.apply_updates(model, updates)` is flagged even though it is right. Fixing those means `cast()` at the boundary, or a local protocol — worth it where the boundary is one we cross often, noise where it isn't.

So the question to answer before switching it on is whether the findings cluster at a handful of library boundaries (worth annotating once) or scatter (not worth it). If it's the former, enable the rule and `cast()` at those boundaries; if the latter, leave it off and note here that we looked.

## Notes

**2026-09-15, tech debt** — Answered: both, and the split runs along a line worth keeping. Of the 29 findings, 14 scatter across `src/mini/`, `scripts/` and `src/utils/` and are one-line fixes at the boundary that produced the `Any` — a `re.Match` group, `json.loads`, a `getattr`, a kwargs bag, `dis.Instruction.argval`, a dynamically imported module. Those are fixed and the rule is on as an `error` in `[tool.ty.rules]`. The other 15 fall into two groups that each have a reason to stay off, both written into `[[tool.ty.overrides]]` beside the rule.

The `docs/` four are the house notebook convention, not mistakes. A report's first binding out of an untyped dict carries the annotation deliberately — `epochs: int = c["epochs"]` — because Marimo copies it onto every downstream cell signature, which is what the `style-py` skill teaches. The rule reads that line as an unsound assignment, so switching it on for reports would argue with the convention on every notebook. Off there for good, unless the conventions move.

The `src/sca/` eleven cluster exactly as this item hoped — three `equinox` entry points (`apply_updates`, `nn.inference_mode`, `filter_checkpoint`) plus a handful of our own methods missing a return annotation — and the fix is small and was written. What stopped it was the price, which this item didn't anticipate: mini keys a memoized task on the *source text* of every project helper it reaches (`_collect_sources` → `_without_docstrings(inspect.getsource(fn))`), so an annotation-only edit re-fingerprints the callers, and a stale DONE record re-runs. Measured: 76 experiment functions across 21 experiments move, including every `train_one` in M2. Deferred to `ty-unsound-assignment-in-sca`, which carries the diff and the cost.

Worth recording that the rule paid for itself on its first run: with the `eqx.apply_updates` boundary typed, ty immediately found that `make_anchored_train_step` promises to accept any `LanguageModel` and then hands the model to `clean_embedding_rows`, which needs an `NGPT`. The `Any` had been hiding it. Details in the follow-up item.

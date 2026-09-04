---
status: open
tags: [docs]
---
# Docs rework

Was tracked as [#45](https://github.com/z0u/mi-ni/issues/45), closed 2026-07-03 after [#48](https://github.com/z0u/mi-ni/pull/48) took the first slice: the index reworked into a narrative arc, the orphaned `role-demo` removed, and the README's missing detached-flow bullet added. The closing audit left two things open, recorded here so the closed issue isn't the only place they live: whether `docs/README.md` should absorb a "how to add a notebook" recipe, and the `gpt.py` / `gpt-sweep` overlap (same models, same architecture story, distinct feature narratives), deferred until one of them next needs substantive edits.

Touches `docs/`, `README.md`, `eng/`, not `src/mini/`. Can run in parallel with anything.

## Notes

**2026-09-04, backport** — sca2's audit of the template experiments ([`remove-template-experiments`](https://github.com/z0u/sca2/blob/main/todo/eng/remove-template-experiments.md) there) is worth reading before touching `docs/pipeline`, `docs/acts` and `docs/probe`: the e2e tests, the CLI help strings, `eng/artifacts.md` and the storage reference all lean on them, and `docs/pipeline/experiment.py`'s module docstring is a page of onboarding prose about `--watch`, wake-at-a-time driving, and content memoization that has no other home yet.

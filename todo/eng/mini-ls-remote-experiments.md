---
status: open
tags: [cli]
opened: 2026-07-14
bundle: cli-devx
---
# `mini ls` can't enumerate the experiments that exist on Modal

From the 2026-07-14 cold-exploration session on CLI usability; the copy-pasteable-hints / sorting / help-text tier shipped (see [mi-ni#57](https://github.com/z0u/mi-ni/issues/57) for the running thread).

`mini ls` reads local launch state only and (alone among the verbs) has no `--app` — there's no way to enumerate experiments that exist on Modal; you must already know the name. The empty-state hint now says so, but listing would be better.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/mini-ls-remote-experiments.md`](https://github.com/z0u/sca2/blob/main/todo/eng/mini-ls-remote-experiments.md) there) with the code it describes.

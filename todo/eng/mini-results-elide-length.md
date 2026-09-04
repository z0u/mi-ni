---
status: done
tags: [cli]
opened: 2026-08-10
closed: 2026-08-10
bundle: cli-devx
---
# `mini results <name>` dumped ~120 KB of floats

From the 2026-07-14 cold-exploration session on CLI usability; the copy-pasteable-hints / sorting / help-text tier shipped (see [mi-ni#57](https://github.com/z0u/mi-ni/issues/57) for the running thread).

It now walks the result and elides length only — a long sequence keeps its first few elements plus a count, an artifact shows name/size/file-count instead of 64-character shas and every child blob, arrays show their shape. Keys and scalars are verbatim and never rounded, so the summary can be read as the result; `--full` gives the repr. An ex-2.1.10-shaped result: 56 KB → 539 characters. `--json` was passed over rather than deferred: results carry `Artifact`s and numpy arrays, so a JSON mode needs an encoding convention for them, and that belongs with a gather API rather than a print verb.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/mini-results-elide-length.md`](https://github.com/z0u/sca2/blob/main/todo/eng/mini-results-elide-length.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

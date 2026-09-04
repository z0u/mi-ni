---
status: done
tags: [memoization, docs]
opened: 2026-08-27
closed: 2026-08-30
---
# A documentation-only edit in `src/` re-runs the DAG

The code fingerprint is built from raw `inspect.getsource` text (`mini.memo._collect_sources`, `_collect_class`), so a docstring counts as evidence. Editing one on a task function — or on any project function or class it transitively references — moves the fingerprint and re-runs the task in place, though nothing about its behavior changed.

Verified rather than inferred: fingerprinting a two-function module with `_code_fingerprint`, rewording only the helper's docstring, moves the caller's fingerprint; restoring the original wording restores the original hash. `memo.py` does have an `_is_docstring` predicate, but it serves the module-level import-time scan (`_module_index`) and never reaches the source text that goes into the manifest.

That makes improving documentation in `src/` cost a sweep, which is the wrong price for it — and the cost is invisible at edit time, since nothing warns and the re-run only surfaces on the next tick. [Published sweeps are one tick away from a full re-run](published-sweeps-one-tick-from-rerun.md) is the neighbouring item; the trigger there is a change to the evidence *scheme*, here it is ordinary prose. Its warning applies unchanged, and is the sharper half of the problem: a re-trained sweep may not reproduce numbers a published report already quotes.

This has almost certainly already happened once. [Un/rewrap all multiline prose strings](unwrap-multiline-prose-strings.md) rewrapped docstrings across `src/` mechanically, and verified prose fidelity, rendered HTML, and the test suite — none of which would show a re-stamped fingerprint. Whether any experiment was actually re-run afterwards is unchecked; only that the evidence would have moved.

The fix is small and the semantics are clean, since a docstring is the one piece of source with no behavior behind it: strip docstrings from each collected source before hashing, by parsing with `ast` and blanking the docstring nodes' line ranges (`_is_docstring` already identifies them, and `_segment` already does line-range slicing). Comments are the same in principle and worse in practice — a changed comment usually accompanies changed code — so leave those in and keep the over-invalidating bias there.

Worth weighing against doing nothing: over-invalidation is the right default, and this is a narrow carve-out. But it is a carve-out that makes the documentation pass safe, and this repo does documentation passes.

## Notes

**2026-08-30, closing** — Fixed as described: `mini.memo._without_docstrings` parses each collected source and blanks the docstring nodes' line ranges, applied at the five sites where source text enters the manifest (`_collect_sources`, `_collect_class`, `_whole_module`, and both branches of `_resolve_ref`). Comments stay in, so the over-invalidating bias holds everywhere else. `tests/mini/test_fingerprint.py::test_docstring_edits_do_not_invalidate` pins all four cases: a function docstring and a module docstring leave the evidence still, a comment and a code edit still move it.

Two things the item didn't say, worth recording. The helper is for source only — a value's JSON encoding must not go through it, since a value encoding to a bare string parses as a module whose only statement is a docstring and would blank to nothing. And the neighbouring item's warning is milder than it reads here: a report reads published refs from the store (`store.get_refs`), never the DAG, so the re-stamped evidence costs nothing until someone deliberately ticks an experiment. The landing order still mattered — this went in ahead of the symbol-normalization pass over `src/` docstrings, so that pass was free.

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/docstring-edits-move-the-memo-fingerprint.md`](https://github.com/z0u/sca2/blob/main/todo/eng/docstring-edits-move-the-memo-fingerprint.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

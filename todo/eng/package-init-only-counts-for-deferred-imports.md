---
status: done
tags: [memoization]
opened: 2026-08-17
closed: 2026-08-18
---
# A package `__init__.py` is evidence for deferred imports only

`_package_chain` folds each parent package's whole source into the manifest, so a deferred `from sca.compute.geometry import probe_maps` depends on `sca/__init__.py` and `sca/compute/__init__.py`. That is the right call — they execute on the way down, and ours sets `XLA_FLAGS`, which changes what the task computes. The live-object walk has no equivalent: a task that imports at module top reaches the helper as an object, `_collect_sources` records the function and its references, and no package source enters. Verified by fingerprinting the two shapes against an `__init__.py` that gained a definition — the deferred one moved, the top-level one didn't.

Low impact today, because task bodies in this project import deferred to keep the driver and CLI light, which is the covered path. It's the same class of hole as the unresolvable-module one though: a silent stale hit rather than a spurious re-run, so it's worth closing before some future task imports its helpers at module scope.

The fix has an asymmetry worth thinking through first. The deferred walk knows the dotted module name it is resolving, so the chain falls out of the name; the live walk starts from a function object, where `fn.__module__` is available but the packages above it have already been imported in the driver process, and folding their whole source in would invalidate more broadly than the deferred path does for the same code (the driver imports far more than any one task reaches). Narrowing to the prelude — the import-time statements, which is what actually differs between processes — may be the better shape for both walks.

## Notes

**2026-08-18, closed** — `_import_time_chain` reads the chain off `__module__` and feeds it into the same work list the deferred walk uses, so both shapes now fold in the parent packages whole plus the defining module's prelude. Two parametrized tests in `test_fingerprint.py` run each shape through the same assertions; both module-scope cases fail without the change and both deferred cases pass either way, so the tests pin the fix rather than the status quo.

The breadth worry didn't survive measurement. The live walk only reaches objects the task actually references, not everything the driver imported, so the chain it adds is the same one the deferred walk would add for the same helper. Fingerprinting ex-2.1.5's four tasks before and after moved nothing at all — 46/48/68/6 manifest entries either way, byte-identical — because those tasks import deferred and already carried `module:sca`. So no published sweep is a tick away from re-running on this ([published-sweeps-one-tick-from-rerun](./published-sweeps-one-tick-from-rerun.md)).

Prelude-narrowing is the part I left alone, and I'd argue against it now. `prelude` holds only the top-level statements that bind no name, so it misses import-time behavior attached to something that does: `CONFIG = _load_and_set_env()` is an `Assign`, and `@register`-decorated defs run their decorator on import — both land in `defs`. Narrowing the chain to it would trade a bounded over-invalidation for a silent stale hit, against the bias the walk is built on. Widening "prelude" to cover those cases converges on the whole module anyway. The one real cost of the whole-file fold is that adding a re-export to an `__init__.py` re-runs everything beneath it; if that starts to bite, the cheaper answer is keeping those files thin, which the reference doc already asks for.

A live object whose module resolves to no file — `__main__`, a notebook cell module, an `exec`'d string — is skipped rather than warned about, since there's no written-down name that should have resolved. That also keeps production experiment modules quiet: `load_experiment` registers them as `mini_experiment_<stem>` and never puts their directory on `sys.path`.

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/package-init-only-counts-for-deferred-imports.md`](https://github.com/z0u/sca2/blob/main/todo/eng/package-init-only-counts-for-deferred-imports.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on; the `sca.compute.geometry` example and the ex-2.1.5 measurement are that project's.

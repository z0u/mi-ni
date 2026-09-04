---
status: done
tags: [memoization]
opened: 2026-07-26
closed: 2026-08-16
---
# An unresolvable module leaves no evidence and says nothing

Deferred-import evidence is now symbol-granular, so the blast-radius half of this is done. What's left is the failure direction that can actually serve a stale hit: if the `sys.path` search doesn't find a module, `_module_index` returns `None` and the walk moves on — indistinguishable from the stdlib and site-packages, which are meant to be skipped. A task importing something the driver process can't see would then depend on nothing and cache forever. Fixing it means telling "deliberately excluded" from "expected to resolve and didn't", which needs a notion of what should have been findable (an installed-distribution check, or a project-roots list). A warning would be enough. Related smaller assumption: `sys.path` order is taken as stable within a process.

## Notes

**2026-08-16, tech-debt session** — Done as the warning. `_should_have_resolved` decides by the *root* package, which is what says whose code a name is: if the root resolves to a project file, a missing leaf under it is a hole (`from sca.thing import x` after `thing` moved); if the root resolves to nothing at all, `sys.stdlib_module_names` and `importlib.metadata.packages_distributions()` are what separate a C extension from an absence. Installed metadata is only consulted once a path search has already failed, so the common path is unchanged. Detection-only by design — no fingerprint, key, or evidence moved, so no existing record was invalidated. Checked for false positives by fingerprinting all 174 task-level functions across every `docs/*/experiment.py`: silent. The `sys.path`-stability assumption is untouched; it stays documented on `_module_file`'s docstring, which is where the caching that relies on it lives.

**2026-08-17, review of [sca2#102](https://github.com/z0u/sca2/pull/102)** — Three benign shapes do reach the warning, reproduced against the merged logic: a project-local PEP 420 namespace package, an optional dependency behind `try/except ImportError`, and a `TYPE_CHECKING`-only import of a package absent at runtime. None occur in this repo, so nothing here reopens; picked up as [fingerprint-warning-false-positives](./fingerprint-warning-false-positives.md).

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/unresolvable-module-leaves-no-evidence.md`](https://github.com/z0u/sca2/blob/main/todo/eng/unresolvable-module-leaves-no-evidence.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

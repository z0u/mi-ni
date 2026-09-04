---
status: done
tags: [memoization]
opened: 2026-08-17
closed: 2026-08-17
---
# Three shapes make the unresolvable-module warning fire on healthy code

`_should_have_resolved` (added by [sca2#102](https://github.com/z0u/sca2/pull/102), closing [unresolvable-module-leaves-no-evidence](./unresolvable-module-leaves-no-evidence.md)) warns when a name resolves to no source and is neither stdlib nor an installed distribution. Three benign shapes land in that gap. All three were reproduced against the merged logic, so this is measured rather than predicted; none of them occur in the repo today, which is why [sca2#102](https://github.com/z0u/sca2/pull/102) shipped.

A **project-local PEP 420 namespace package** — a directory on `sys.path` with no `__init__.py` — warns on the package itself. The leaf is tracked correctly (`nspkg.leaf:go` reaches the evidence), but `_package_chain` also yields the bare package as a reference, and there is no `__init__.py` for `_module_file` to find. The installed-metadata check covers namespace packages that were installed; a local one has no metadata to consult. This is the one with a cheap fix: `_should_have_resolved` could return `False` when some `sys.path` entry holds a *directory* named for the root, which is what a namespace package looks like from outside, and `_module_file` already walks those entries.

An **optional dependency behind `try/except ImportError`** warns whenever the dependency is absent, which is the case the code was written to handle. Same for an **`if TYPE_CHECKING:` import** of a package not installed at runtime, which cannot affect what the task computes. Both need the AST context that says "this import is guarded", and by the time the walk reaches them it is reading bytecode, where the `try` and the `if` have become jumps. Documenting the two is the realistic answer.

The doc entry in `.agents/skills/mi-ni/references/memoization.md` currently reads as though any warning means a missing install, and the message points the reader at the editable install and `PYTHONPATH`. That remedy is wrong for all three, so whichever way this is settled, the message and the entry should admit that a warning can also mean "deliberately absent".

Worth weighing against the reason the stdlib carve-out exists at all: a warning that fires on the ordinary case stops being read, and this one guards a failure that costs results.

## Notes

**2026-08-17, tech-debt session** — Settled the way the item proposed: the namespace package is fixed, the two guarded shapes are documented. `_is_namespace_portion` names the *portion* rather than its root, so `nspkg` goes quiet while `nspkg.gone` still reads as the hole it is — the narrower test than the item suggested, and it keeps the missing-submodule warning that the check exists for. The message and the `memoization.md` entry now both say a warning can mean "deliberately absent", so neither sends a reader after an install that was never broken. Three tests: the namespace silence, the hole under it, and the `try/except ImportError` shape pinned as known behaviour — that last one should flip if a later change learns to read the guard.

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/fingerprint-warning-false-positives.md`](https://github.com/z0u/sca2/blob/main/todo/eng/fingerprint-warning-false-positives.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on. `memoization.md` here carries the reworded entry.

---
status: done
tags: [tooling]
opened: 2026-08-06
closed: 2026-08-14
---
# `mini/__init__.py` re-exports the whole package

`from mini.reports import export_key` runs `apparatus`, `modal_apparatus`, `experiment`, `store`… so a leaf module with only stdlib imports still needs the full environment. Cost a round in CI: `scripts/unpublished_reports.py` was written to run before the install and couldn't. Not urgent — nothing else wants a lightweight import today — but worth remembering before the next standalone tool.

The re-exports are now lazy (PEP 562 module `__getattr__`), resolving through a name→submodule map and caching into the module namespace on first touch. `from mini import Ctx` reads the same; `from mini import <submodule>` still works, because the import system falls back to importing the submodule when `__getattr__` raises `AttributeError`.

Three declarations now describe one set — the `TYPE_CHECKING` import block (what `ty` and IDEs see), the map (what resolves at runtime), and a literal `__all__` (ruff and `ty` read only the literal form, so a computed one silently turns off F401/F822). `tests/mini/test_lazy_exports.py` asserts all three agree, so a name added to one and not the others fails a test rather than an import.

Measured whole-process, best of 5: `import mini.reports` 346 ms → 55 ms, `python -m mini` 333 ms → 145 ms — the CLI keeps what it genuinely uses but no longer pays for `modal` unless `--app modal` asks for it. `scripts/unpublished_reports.py` now runs to completion on a bare `/usr/bin/python3` with no venv and no dependencies (it raised `NameError` from the eager chain before), so the publish check *could* move ahead of the install step in `lint-check.yml` and stop being gated on `steps.install.outcome`. Left alone here: that reorders the job summary, which the workflow comment argues for deliberately, so it wants its own decision rather than riding along.

Two of vulture's `ignore_names` came with it (`__getattr__`, `__dir__`): interpreter-called hooks with no caller to find, alongside the marimo entries already there.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/mini-init-reexports-everything.md`](https://github.com/z0u/sca2/blob/main/todo/eng/mini-init-reexports-everything.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

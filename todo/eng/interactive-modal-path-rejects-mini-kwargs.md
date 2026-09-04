---
status: open
tags: [library, modal]
opened: 2026-09-03
bundle: inherited-from-sca2
---
# The interactive Modal path rejects mini's own kwargs

`_build_modal_fn` (`src/mini/modal_apparatus.py`) pops only `startup_timeout`, while the memo path also drops `watchdog`, `watchdog_grace`, and `name`. So `.w(watchdog=600)` followed by `map` raises a TypeError from Modal. Not reachable from the CLI, which goes through the memo path; only a notebook that drives the `Apparatus` by hand sees it.

Found while reviewing the ported `src/mini` in the 2026-09-03 backport and inherited from sca2's tree, so the fix belongs there first and then back here.

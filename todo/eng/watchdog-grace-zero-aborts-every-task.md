---
status: open
tags: [library, cli]
opened: 2026-09-03
bundle: inherited-from-sca2
---
# `--watchdog-grace 0` aborts every task before its first emission

The stamp in `src/mini/_taskworker.py` treats 0 as unset, but the value still reaches the watchdog, which then has no grace at all. Either 0 should mean "no grace" end to end, or it should be refused at the CLI.

Found while reviewing the ported `src/mini` in the 2026-09-03 backport and inherited from sca2's tree, so the fix belongs there first and then back here.

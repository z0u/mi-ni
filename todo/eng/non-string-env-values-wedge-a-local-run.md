---
status: open
tags: [library, cli]
opened: 2026-09-03
bundle: inherited-from-sca2
---
# Non-string `[tool.mini] env` values wedge a local run

The subprocess spawn (`src/mini/local_apparatus.py`) raises on an int after the record was already claimed RUNNING with no pid, so reap never settles it and only `cancel` clears it. Modal rejects the same config with a clean error; the local backend should validate up front too, before it claims anything.

Found while reviewing the ported `src/mini` in the 2026-09-03 backport and inherited from sca2's tree, so the fix belongs there first and then back here.

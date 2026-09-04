---
status: open
tags: [memoization]
opened: 2026-09-03
bundle: inherited-from-sca2
---
# Property bodies aren't walked by the memo fingerprint

`_collect_class` (`src/mini/memo.py`) recurses into plain functions, staticmethods, and classmethods, so a deferred import inside a `property` or `cached_property` leaves the memo key unchanged. The one place the "deferred imports are traced" claim in the memoization reference doesn't hold; the fix is to unwrap those descriptors to their `fget` on the way down.

Found while reviewing the ported `src/mini` in the 2026-09-03 backport and inherited from sca2's tree, so the fix belongs there first and then back here.

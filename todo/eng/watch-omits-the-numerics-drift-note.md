---
status: open
tags: [cli]
opened: 2026-09-03
bundle: inherited-from-sca2
---
# Plain `watch` omits the numerics-drift note

`status` and `watch --json` both carry the note (`src/mini/__main__.py`); the plain `watch` view doesn't, so the one reader who is looking at a run live is the one who doesn't see it.

Found while reviewing the ported `src/mini` in the 2026-09-03 backport and inherited from sca2's tree, so the fix belongs there first and then back here.

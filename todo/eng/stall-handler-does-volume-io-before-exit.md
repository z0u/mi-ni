---
status: open
tags: [library, modal]
opened: 2026-09-03
bundle: inherited-from-sca2
---
# Unverified: the stall handler does Volume I/O before the hard exit

If that I/O blocks on a wedged container, the exit never runs, and the stall the handler exists to report becomes a hang. Worth a bounded timeout or an exit-first ordering. Unverified from the code alone: it needs a container that actually wedges to say whether the I/O can block there.

Found while reviewing the ported `src/mini` in the 2026-09-03 backport and inherited from sca2's tree, so the fix belongs there first and then back here.

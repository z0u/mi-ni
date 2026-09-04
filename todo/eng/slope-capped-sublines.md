---
status: open
tags: [vis]
opened: 2026-07-21
---
# Slope-capped sublines

`Sparkline._create_path_data` takes its curve knots from glyph ink bounds, so a ramp is always one inter-glyph gap wide however big the jump is. A large step in surprisal therefore renders near-vertical, which reads as a discontinuity and gives up the rate-of-change cue the smooth step exists for. Deriving the ramp width from the jump height instead (cap the on-screen angle, then shrink adjacent ramps so a plateau survives) fixes it, but it restyles figures already published downstream (sca2's ex-2.1.1 and ex-2.1.2), so it wants an opt-in parameter and a deliberate pass rather than a drive-by edit.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/slope-capped-sublines.md`](https://github.com/z0u/sca2/blob/main/todo/eng/slope-capped-sublines.md) there) with the code it describes.

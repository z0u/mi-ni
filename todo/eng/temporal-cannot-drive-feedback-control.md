---
status: open
tags: [library, temporal]
opened: 2026-07-14
---
# `mini.temporal` can't drive feedback control

`DynamicProp.set()` retargets mid-flight from the current (value, velocity) state — exactly what a controller needs — but experiments consume schedules via `realize_timeline`, which bakes the dopesheet into a static per-step array before training, and the dopesheet's own keyframes would fight any runtime `set()` calls on the same prop. If feedback-driven schedules become standard, consider a Timeline mode where a prop is declared "controlled": keyframes set its *bounds/defaults* and a callback supplies the live value.

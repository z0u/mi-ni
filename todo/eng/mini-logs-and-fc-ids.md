---
status: open
tags: [cli]
opened: 2026-07-14
bundle: cli-devx
---
# `mini logs` holds only failure tracebacks, and the `fc-…` ids feed nothing back

From the 2026-07-14 cold-exploration session on CLI usability; the copy-pasteable-hints / sorting / help-text tier shipped (see [mi-ni#57](https://github.com/z0u/mi-ni/issues/57) for the running thread).

`mini logs` holds only failure tracebacks (the help text now says so), and the Modal `fc-…` ids that `status` prints can't be fed back into any `mini` verb — worker stdout/logs need the Modal dashboard.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/mini-logs-and-fc-ids.md`](https://github.com/z0u/sca2/blob/main/todo/eng/mini-logs-and-fc-ids.md) there) with the code it describes.

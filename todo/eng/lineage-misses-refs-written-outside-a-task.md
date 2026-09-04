---
status: open
tags: [library, storage]
opened: 2026-07-14
---
# Lineage is auto-detected, except for refs written outside a task worker

Cross-experiment lineage is detected without annotation: `set_ref` in a task worker stamps producer identity onto the ref, `get_ref` records the resolution on the task record (`upstream_refs`), and the driver rolls both into `lineage.upstreams`.

Known gaps: refs written by the interactive `Apparatus` (`app.map` in a notebook) or by driver-side code are unstamped, and a consumer served entirely from memo hits records nothing new — its previously-recorded `upstream_refs` persist on the old records, which is usually what you want. Pre-existing refs stay unstamped until their publish step re-runs.

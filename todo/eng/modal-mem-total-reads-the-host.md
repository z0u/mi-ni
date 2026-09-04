---
status: finding
tags: [library, modal]
opened: 2026-07-14
---
# Modal `mem_total_gb` reads the host, not the container

Modal `mem_total_gb` in a task's `env` reads the *host* total from `/proc/meminfo` (gvisor shows the whole node), not the container's memory limit. Fine as a coarse "what class of machine" signal. If we ever want the true per-container cap, read the requested `memory=` from the role config instead (or the cgroup limit, if gvisor exposes it).

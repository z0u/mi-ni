---
name: mi-ni
description: How to use the library code provided by mi-ni. Code design patterns that abstract compute to easily scale experiments. Read to learn about the `mini` package, the `Apparatus` class, hyperparameter schedulers, notebook/visualization utils, and how to author, run, and monitor memoized experiments from the CLI. These should be used by default.
---

Library structure:

```
src/mini/
├── apparatus.py  # Base Apparatus class
├── volume.py     # Base Volume class
├── local_*.py    # Apparatus that uses local compute and storage
├── modal_*.py    # Apparatus that uses cloud GPU compute and storage
├── temporal/     # Advanced hyperparameter scheduling based on keyframes
└── vis/          # Visualization helpers
```

## Two ways to compute

- **Interactive `Apparatus`** (`app.map`/`app.arun`) — a blocking call inside a notebook; dies with the process. Use for quick, light work you watch finish.
- **Memoized orchestration** (`Experiment(main=...)`, driven by the `mini` CLI) — detached, durable, pollable across short-lived processes. Use for sweeps, multi-step pipelines, anything slow, and anything an agent runs autonomously.

## Authoring, running, monitoring

- **Author** a memoized experiment — the `main(ctx)` DAG, repo layout, and cache-friendly design: [authoring.md](./references/authoring.md). The identity/evidence model behind caching: [memoization.md](./references/memoization.md).
- **Run & monitor** one from the CLI — the wake-loop, backend routing, wall-clock budget, and how to delegate/schedule a long run: [running.md](./references/running.md).
- **Fix & recover** — the fix/prune/retry loop, bounded hotfixes, superseded records, and partial `map` failures: [recovery.md](./references/recovery.md).
- **Store & share large outputs** — return `Artifact` handles instead of volume paths, share artifacts across experiments by name, and publish artifacts to a URL: [storage.md](./references/storage.md).
- **Publish a report to the web** — externalize a report's figures and data into a bundle: [reports.md](./references/reports.md); the `themed` figure hook that feeds it: [vis.md](./references/vis.md).

To keep cost down, delegate launching and babysitting to the `experiment-monitor` subagent (it escalates to `experiment-doctor`); see running.md.

## Apparatus and Volume

`mini` provides experiment infrastructure via the `Apparatus` class. Its interface is similar to an Executor, but it abstracts compute and storage. See [apparatus.md](./references/apparatus.md), especially if you're using it in a notebook.

## Hyperparameter scheduling

`mini.temporal` provides advanced hyperparameter scheduling. An experiment may define the schedule using a "dopesheet" (table of keyframes); this is then interpolated to a timeline with one value per parameter per step. Useful for adjusting learning dynamics over the course of training.

## Visualization helpers

`mini.vis` provides utilities for figure theming (base styles, light/dark support). See [vis.md](./references/vis.md). Externalizing those figures and a report's data assets so the exported HTML stays light and publishes off Git LFS is a report concern: [reports.md](./references/reports.md).

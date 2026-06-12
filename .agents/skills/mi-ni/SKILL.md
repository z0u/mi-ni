---
name: mi-ni
description: How to use the library code provided by mi-ni. Code design patterns that abstract compute to easily scale experiments. Read to learn about the `mini` package, the `Apparatus` class, hyperparameter schedulers, and notebook/visualization utils. These should be used by default.
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

## Apparatus and Volume

`mini` provides experiment infrastructure via the `Apparatus` class. Its interface is similar to an Executor, but it abstracts compute and storage. See [apparatus.md](./references/apparatus.md), especially if you're using it in a notebook.

## Hyperparameter scheduling

`mini.temporal` provides advanced hyperparameter scheduling. An experiment may define the schedule using a "dopesheet" (table of keyframes); this is then interpolated to a timeline with one value per parameter per step. Useful for adjusting learning dynamics over the course of training.

## Visualization helpers

`mini.vis` provides utilities for figure theming, such as base styles and light/dark support. See [vis.md](./references/vis.md).

---
name: style-terms
description: |
  Shared terminology for experiment reports: condition vs. cell, etc.
  Some terms differ slightly from convention, so always use when working on reports.
---

## Methodological terms

Use these terms consistently across all reports.

- factor

  A swept design parameter with named levels ("factor A, the anneal endpoint; levels 50, 70, and 90"). The factorial is the crossing of the factors.

- condition

  One combination of factor levels in a sweep or factorial design; seed-aggregated. "The `warmup-long` condition", "every condition misses on the holdout". Never "cell".[^not-cell]

- run

  A condition crossed with a seed: one training run, the unit of replication. Three seeds per condition means three runs per condition.

- arm

  An extra design parameter outside the factorial (or ladder), branching off one grid condition with one thing changed to answer one question (a timing arm, a ceiling arm, a dose arm). Arms ride along, typically unscored.

- trial

  One sampled point in a survey's search space; seed-aggregated, like a condition. "Condition" implies named levels chosen in advance, which a sampled point doesn't have, so use "trial" wherever the point came out of a sampling rule. Surveys have trials and no arms. See the `science` skill for the experiment type.

- seed

  The replication factor. Prefer "seed mean" / "seed range" for aggregates over runs.

- criterion

  One clause of a hypothesis gate. Calling these "conditions" would collide with the design sense above and produces sentences like "every condition a condition misses".

## Indexing

Math and prose count from 1; code counts from 0. So a report names *the first axis* or e₁ where the code says `AXIS = 0`. Slight preference for "the first basis vector" over "basis vector 1", etc.

An index that counts *steps applied* rather than positions is not an exception, even though it starts at 0: step 0 is the input, before anything has been applied, and gets a name rather than a number in figures.

## Statistical terms

- R² vs r²

  Two statistics that are easily both written R². Write $R^2$ for a probe's held-out coefficient of determination (can be negative; measures a fitted readout), and $r^2$ for a squared Pearson correlation (bounded to [0, 1]; measures proportionality). Say which one in prose on first use.

[^not-cell]: In classical DoE a condition is called a cell, but we can't call it that because other senses of "cell" appear in reports and cannot be renamed away: cells of a table or heatmap ("each cell is the seed mean"), and Marimo notebook cells ("the analysis cells below"). Reports also legitimately use it for spatial grids (color-grid cells, Voronoi cells). So in prose, "cell" never means a condition or a run. In _code_, a stored key like `metrics["cells"]` can keep its legacy name to avoid invalidating memo keys.

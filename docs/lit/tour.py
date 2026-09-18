# title: A tour of literate scripts

r"""
# A tour of literate scripts

This page exercises what a report needs: interpolated numbers, a table built by a comprehension, math, footnotes, admonitions, a cached figure, and an early stop that leaves the rest of the document readable. Cell code is hidden, as it is by default: a report hides its plumbing. A page that is about the code, like the themed-plots one, opts in with `# code: show` at the top of the file.

The file is plain Python, so ruff, ty, and the IDE see every cell. A top-level string is prose (this paragraph is one), and the code between two of them is a cell.
"""

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from mini.lit import memo, stop
from mini.vis import themed
from mini.vis.theme import light_dark


@dataclass(frozen=True)
class Condition:
    name: str
    seeds: int
    mean: float
    spread: float


CONDS = [
    Condition("control", 20, 0.012, 0.004),
    Condition("anchored", 20, 0.531, 0.021),
    Condition("anchored-slot", 12, 0.498, 0.030),
]
GATE = 0.45
best = max(CONDS, key=lambda c: c.mean)

rf"""
## Numbers in prose

The best condition is `{best.name}` at {best.mean:.3f} ± {best.spread:.3f} over {best.seeds} seeds, against a gate of {GATE}. Prose that quotes a value is an f-string, so a format spec does what it does anywhere in Python, and the names are real references that ruff, ty, and go-to-definition see; a helper in a cell can do anything more involved.

## A table from a comprehension

Today a table like this is assembled as a Markdown string in Python. Here the rows are one field, written where the table appears:

| condition | seeds | mean | verdict |
| --- | ---: | ---: | --- |
{"\n".join(f"| `{c.name}` | {c.seeds} | {c.mean:.3f} ± {c.spread:.3f} | {'**pass**' if c.mean >= GATE else 'miss'} |" for c in CONDS)}
"""

r"""
## Math, footnotes, admonitions

The same Markdown extensions as `mo.md`: inline math \(\bar\alpha = \frac{1}{n}\sum_i \alpha_i\), display math

\[
\mathcal{L} = \mathcal{L}_\text{task} + \lambda \, \lVert h - a \rVert^2,
\]

a footnote[^1], and an admonition. This paragraph is a plain raw string, since the math is full of braces; the admonition below quotes a value, so it is a separate f-string (a paragraph that needs both doubles its math braces).
"""

rf"""
/// admonition | A note on the gate
    type: note
The gate of {GATE} comes from the reference experiment, quoted here from the same constant the cells use.
///

[^1]: Footnotes number from one per document, since the whole page is rendered in one pass.

## A cached figure

The slow part of a re-render is usually the figures. `@memo` caches the rendered HTML keyed by the plot function's source and its arguments (arrays included), and remembers the PNGs it wrote, so the next render, even in a fresh process, skips the draw. This one sleeps for a second to make the point.
"""

rng = np.random.default_rng(0)
samples = {c.name: rng.normal(c.mean, c.spread, c.seeds) for c in CONDS}


@memo
@themed(
    name="tour-conditions",
    alt_text="Dots per condition with the gate as a dashed line.",
    caption="**Per-seed values by condition.** The dashed line is the gate; each dot is one seed.",
)
def conditions_figure(samples: dict[str, np.ndarray], gate: float) -> plt.Figure:
    time.sleep(1.0)  # stand-in for an expensive draw
    fig, ax = plt.subplots(figsize=(5, 2.6), layout="constrained")
    for i, v in enumerate(samples.values()):
        jitter = rng.uniform(-0.1, 0.1, v.size)
        ax.plot(np.full_like(v, i) + jitter, v, "o", ms=4, alpha=0.7, color=light_dark("#1a5f8a", "#6ab0d4"))
    ax.axhline(gate, ls="--", color=light_dark("#666", "#aaa"), lw=1)
    ax.set_xticks(range(len(samples)), list(samples))
    ax.set_ylabel("value")
    return fig


conditions_figure(samples, GATE)

"""
## Stopping early

A preregistration is written before its results exist. `stop()` ends execution with a message, and every later field that cannot be evaluated renders as a pending mark instead of an error, so the whole document still reads: headings, paragraphs, and the values that are known.
"""

results = None
if results is None:
    stop("/// admonition | Results to come\n    type: warning\nThe experiment has not been published yet.\n///")
summary = results["summary"]

rf"""
**Results.** The anchored condition reached {summary.m_line} with a lead of {summary.lead}; see {summary.figure}.

This paragraph has no interpolation, so it renders as written.
"""

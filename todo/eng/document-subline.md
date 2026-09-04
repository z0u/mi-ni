---
status: done
tags: [skills, vis]
closed: 2026-08-27
---
# Document subline in a skill

Describe subline in a skill: what it is, why we might use it instead of a token heatmap, and how to use it.

## Notes

**2026-08-27, tech debt** — Landed as a short `Sublines` section in `style-fig`, cross-linked from the "never use heat maps for sequences" bullet so the alternative is findable from the rule. The mechanics went into `Subline.plot` and `Series`, which had one-line docstrings before; the skill keeps only what is ours rather than the library's — the `--bg-color` override for our notebook background, and the `figure_html`/externalize wrapping.

Each mechanic was verified against the code by rendering a probe SVG and reading the path data: values are fractions of the band (0 on the baseline, 1 at the top, negatives silently clipped away, the band itself rendering up to 2 — which is why ex-2.1.1 and ex-2.1.2 both `np.clip(..., 0, 1)`); `NaN` breaks the path, which is what makes the position-0 pad line the series up with the text; and only five series colors are defined.

The first draft described the alignment as per-character throughout, which under-sold it. `plot` takes a sequence of tokens of any width, and `TokenBB.is_wide` exists precisely so a wide token holds its value as a plateau across its glyphs before ramping to the next — the same visual grammar as `smooth_step`. A bare string is the special case that splits into characters.

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/document-subline.md`](https://github.com/z0u/sca2/blob/main/todo/eng/document-subline.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on; the `Sublines` section is in `style-fig` here.

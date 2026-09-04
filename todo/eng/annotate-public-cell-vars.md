---
status: partial
tags: [notebooks, tooling, typing]
opened: 2026-08-12
---
# Require public cell variables to be annotated, with the most specific type available

Marimo propagates a definition's annotation onto the parameters of every cell downstream (see `style-py`), so one annotation at the definition types the whole notebook, and a missing one leaves every consumer as `Unknown`. Two halves, and only the first is mechanizable.

**Annotation present.** Done: `scripts/unannotated_cell_vars.py`, run as `./go annotations [...paths]` and as an advisory CI step beside the dead-code one. For each `@app.cell` function it flags a public name bound by a bare `Assign` rather than an `AnnAssign`; names bound by `def`/`class` carry their own types, and tuple unpacking can't carry an annotation, so those are reported separately and want splitting, or an opt-out. It parses with `ast` rather than ast-grep, and the parse turned out to be its own predicate for "is this a notebook". Advisory rather than a gate because the project it was built in arrived with a backlog of ~90 bare assignments, and each fix has to be published with its report. The gate is the natural next step once a project's list is short: move it into `./go lint`, and add it to `.claude/hooks/marimo-format.sh` so a new one is caught where it's written. One refinement worth knowing: a name that isn't in its cell's `return` tuple propagates nowhere, so it wants a leading underscore rather than a type, and the return tuple is a better filter than the underscore convention for finding the annotations that pay. Left out because it depends on Marimo having re-saved the file, which the edit hook keeps current but the checker would rather not assume.

**Annotation specific.** `dict[int, dict]` and `np.ndarray` pass the first check while saying almost nothing. *Arrays* want jaxtyping, already a dependency and the convention in `src/`: `Float[np.ndarray, "L1 N T C"]` on the analysis side, `Float[Array, "L1 B T C"]` on the JAX side, over one shape vocabulary. Checked 2026-08-12 that Marimo propagates the full annotation verbatim, shape string included, and `ty` and `ruff` both pass on it inside a notebook. Nothing verifies these shapes at runtime (no `jaxtyped`, beartype, or typeguard anywhere), so they document rather than check, and turning runtime checking on is a separate and larger decision. *Records* are where jaxtyping has nothing to say: `dict[int, dict]` wants a `TypedDict` for the metrics/trial/cell records every report indexes by string key, and since those shapes are shared across experiments they belong in `src/` rather than being redeclared per notebook. A weaker mechanical backstop, if we want one: flag unparameterized `dict`/`list`/`np.ndarray`, and select `ANN401` for a literal `Any`.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/annotate-public-cell-vars.md`](https://github.com/z0u/sca2/blob/main/todo/eng/annotate-public-cell-vars.md) there), with the checker; the backlog counts and the per-report breakdown in the original are that project's, so they were left behind.

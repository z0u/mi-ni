---
name: science
description: |
  How we do experimental science in this project: designing experiments, preregistering falsifiable hypotheses, and collaborating on reports.
---

## Preregistration

Design the experiment with the human, and draft the report skeleton before writing any experiment code. The skeleton doubles as the analysis plan: writing it before the data exists lets a later "we predicted X and found Y" carry weight, because the prediction is verifiably older than the result.

The skeleton is usually a text-only notebook. It runs: intro (the question, why it matters, lineage from earlier experiments), a `Findings` section left empty until the results land, a "How to read this draft" note, the method (data spec, measurements), the hypotheses with decision thresholds, analysis sections (consider having one section per hypothesis), an "Exploratory analyses" section, and a discussion.

Conventions:

- Placeholders are admonitions marked `TODO`. Each states what its figure or table will show (axes, panels), the hypothesis it scores, the expected pattern, and what a contrary result would look like. The marker is greppable, so no placeholder survives to publication; results replace placeholders in place, so review reads as a prediction → observation diff.
- Hypotheses are falsifiable: state the measurement, the threshold, and which outcomes count as partial.
- A constants-only `experiment.py` is marked `DESIGN_ONLY = True`. Landing the design constants — grid sizes, thresholds, schedules — as a module during preregistration lets the report import them instead of restating numbers the code will later own. But `tests/mini/test_experiments_e2e.py` globs every `docs/**/experiment.py` and asserts it loads into a named experiment with a callable `main(ctx)`, which a design module doesn't have yet; `DESIGN_ONLY = True` at module level skips that check. Delete the line in the same change that adds the DAG, or the implemented experiment silently loses its load coverage.
- Freeze the hypotheses once the skeleton is agreed (immaterial edits aside), and say so in the report under "How to read this draft": results replace placeholders, and anything conceived after seeing the data goes under "Exploratory analyses", marked as post hoc.
- Avoid over-claiming in the analysis and discussion. An experiment may _inform_ the next, but committing to an interpretation now can close off the follow-up.
- A claim stated before its evidence exists gets paid for twice, once where it is stated and once where it is met. That is inherent to preregistration and worth the cost for hypotheses and thresholds, and not for anything else, so keep rationales, caveats, and worked reasoning at the point of use rather than in the method. Where a restatement is unavoidable, quote the frozen line rather than paraphrasing it, since a paraphrase drifts.
- Numbers in prose earn their place by being part of an argument. A coordinate the reader looks up, a constant of the apparatus, or a value derivable from an adjacent table belongs in a table or in the method, with the prose referring to it. Writing the same quantity out in two sections is how two roundings of it end up in the report.

Example:

```md
## Hypotheses

- **H1.** Describe what we're testing (no title).

<!-- Then in the results/analysis section further down... -->

## Short name for H1 (H1)

/// admonition | TODO
Describe what is needed (figure, table, expectations).
///
```

## Surveys

Some questions are about choosing an operating point in a space too large to give every point a hypothesis: a regularizer's weight, a temperature, how much of an intervention to apply and when. A *survey* is the experiment type for those. It preregisters the search plan instead of an outcome, and it scores nothing.

What makes a search credible is the same thing that makes preregistration work — the analysis was fixed before the data existed. So freeze the procedure, in place of `## Hypotheses`:

- **The space.** Each dimension with its bounds and its scale (log for weights and temperatures).
- **The sampling rule**, with its seed, so the trial list can be reviewed before the run. Prefer Sobol (`scipy.stats.qmc`) over uniform draws: it fills the box more evenly, so the one-dimensional marginals are less lumpy for the same budget. SciPy only reaches us through scikit-learn, so declare it when the first survey lands.
- **The trial budget**, and the `--budget` and `--max-containers` caps that hold it.
- **The objective.** Where objectives trade against each other — the usual case, since the knob that buys one thing tends to spend another — state it as a constraint ("maximize the effect subject to holdout accuracy staying within `TASK_GATE` of control") and report the Pareto front rather than a scalar winner.
- **The seed budget:** how many seeds per trial in the first round, which band gets promoted, and to how many. Cut cost on the seed axis rather than by stopping runs early, unless the objective is known to be monotone in training time: where it peaks mid-run and drifts afterwards, a short-run proxy favours configurations that look good early.
- **The noise floor of each objective**, measured first, and the resolution it licenses. A survey may not claim a difference it cannot resolve. Sometimes this is free: a past condition with many seeds gives the per-run spread of every statistic at that operating point, already in the store.
- **The stopping rule.**

With those fixed, everything the survey reports is a deterministic function of the data, so there is no forking-path problem. The only freedom left is which point wins, and that is the output rather than a claim.

Two rules make the type safe to publish.

**Nothing a survey reports may be quoted as a result.** It proposes an operating point; the next preregistered experiment adopts that point, scores it at fresh seeds, and reports the survey's value beside the confirmed one. The gap is the winner's-curse correction — a search's best trial wins partly on merit and partly on lucky seeds, so re-measuring is what turns a proposal into a number. That handoff already happens informally, whenever a follow-up runs at the operating point an earlier experiment landed on; naming it makes the proposing half publishable.

**Publish every trial**, including the ones that went nowhere. Selective reporting is what would make a large search worthless, and a complete table settles it. Memoization means the data is there anyway.

Then report the landscape rather than the winner. "The effect holds above 0.5 for the weight anywhere in [0.05, 0.4]" is worth more than "0.12 was best": it is what the next experiment inherits, and a wide plateau is itself a result, since it says the method does not need careful tuning.

### A survey's report

Same skeleton, with three differences.

- `## Findings` becomes `## Observations` — same place, same brevity, but each line carries its noise floor where a verdict would carry its gate, and one line names the proposed operating point.
- `## Hypotheses` becomes `## Search plan`.
- `### Conditions` becomes the space specification plus the full trial table. This is the convention that has to bend: elsewhere the report imports hand-justified condition dicts and renders them as prose, which is why there's no generic grid builder, and a hundred trials can't each carry a docstring. So the justification attaches to the dimension rather than the level, and the trial table is generated from stored results.

Say "survey" in the first clause of the tl;dr and label it the same way in `docs/index.md`. Naming and numbering follow the usual experiment sequence.

`docs/gpt-sweep/report.py` is the closest existing example — a grid over width, learning rate and architecture, no hypotheses, and a conclusion about whether the region is safe to build on. A survey is that plus the frozen search plan, which a small grid didn't need.

## Best practices

- Choose a measurement site by a criterion independent of the statistic you're judging.
- Ablate before you search, and **bracket rather than survey** when the ablation needs a value you haven't found yet. Removing a schedule means running a constant instead, and which constant you pick can decide the answer. Rather than matching on one invariant (area, maximum, or endpoint — each defensible, each a different condition), choose flat levels that straddle the schedule's own range. If the schedule beats every constant in its range, no constant substitutes for the shape, whatever the optimum turns out to be; if one wins, the schedule dimensions go away and you have a better operating point too. Report the matching invariants for every arm instead of matching on one, so the results can say which was the active ingredient. Each dimension deleted this way is much cheaper than searching it.
- (more in [/todo.md](/todo.md))

## Collaborating on a report

The human wants to be involved in the writing, so the skeleton is a review artifact in its own right. Iterate on it together in a PR before any experiment code lands (although feel free to run small prototypes that don't get committed). This is where the hypotheses and thresholds get agreed and frozen.

When results arrive, fill the report in order of stakes rather than all at once. The mechanical sections, where the number either clears its threshold or it doesn't, can be filled in one pass. Pause for a discussion round before writing the prose where interpretation lives, since that is the part the human most wants a hand in, and the part most likely to over-reach.

Whatever you have written, run a review round over it before handing back to the human, covering the sections that are done. Say in the request which sections are in scope, so a `TODO` in a section whose turn hasn't come isn't read as an omission.

Any prose you write gets two passes on the same turn, whether or not a review round is warranted: `prose-simplifier` to lower reader effort, then the `report-restructure` skill to give the result a shape that can be skimmed. Run them in that order: the simplifier makes dense sentences parseable, and the restructure pass then groups them and cuts what repeats. Stage your changes first so you can see what each pass did, then read the edits for correctness. Both get the path and line range and nothing else. This applies to a single filled-in section as much as to a whole draft, so treat it as a habit of writing rather than a step in the review. The sequence, and the checks each pass leaves to you, are in [references/review-passes.md](references/review-passes.md).

Neither pass will make a report much shorter: in a section that is 40% figure captions, alt text, and tables, all three are protected, so there is little left that can move. Length comes off at the structural level instead — duplication across sections, and front matter that runs before the first result — which is the `report-structure` agent's job at the freeze and publish gates.

The publishing mechanics — exporting the report as a bundle, wiring result refs, verifying the render — are a separate concern, covered by the `mi-ni` skill.

### The tl;dr

A report opens with one, directly under the title and above the intro prose:

```md
# A wider model and a longer warmup

/// tip |
<!-- tl;dr -->
We tested two changes aimed at stabilizing training:
**1.** Double the model width, and
**2.** Extend the learning-rate warmup.
Both work, but 2. worked better, and their effects stack.
///
```

Four lines or so (two when reflowed): what we tried, and which way it came out. It orients someone deciding whether to read on, so keep numbers, hypothesis IDs, thresholds, and caveats out of it. The analysis sections and the discussion carry the full accounting. Left to itself this box grows into a second conclusion; if a sentence in it would also belong in the discussion, cut it.

The title is empty (`/// tip |`) so the box reads as a lede rather than a labelled aside, and the `<!-- tl;dr -->` comment keeps the marker greppable. In a preregistration draft, write the "what we tried" half and leave the outcome line for later.

### Findings

Directly under the tl;dr, and above the intro prose. Every preregistered hypothesis, its verdict, and the one number that decides it, with its gate inline so the section stands alone. Under 200 words:

```md
## Findings

**H1 (task cost) — holds.** Largest holdout accuracy gap from control,
across all seven conditions: 0.0013. Gate: 0.02.

**H3 (attribution) — fails.** Both main effects clear 0.1, but the
second effect (+0.141) is smaller than the first (+0.221), not larger;
the ordering holds within every seed.
```

The tl;dr says which way it came out; this says what happened. A reader who stops here should be able to tell that three of four hypotheses missed, without reading a discussion to find out. Without it, a reader gets nothing until they have read the whole report.

Verdicts only. Interpretation, mechanism, and whether an outcome was named in advance belong to the analysis sections. In a preregistration draft the heading goes in empty, since writing it is the first thing to do when results land.

Two consequences for the rest of the report. The discussion no longer opens by re-deriving the results, because they are above it — it interprets, and nothing else. And a section that cannot supply its own line here has a gap worth fixing: if a verdict or its deciding number is missing, or first appears in some other section, that is the section's problem rather than the summary's.

### Recording a review decision

Reports go through several fresh-eyes review rounds, each reader starting from the report alone. `REVIEW` notes are how one round's decisions reach the next.

**Leave a `REVIEW` note wherever a review changes a claim:** a threshold, a verdict, a scope, or a wording that changes what is being asserted. Typos and prose polish don't need one. Say what the change was, why, and what a later reader should check to disagree with it. Usually a Python comment in the cell:

```python
# REVIEW: narrowed "the effect transfers" to "transfers at layer 4" — H2 only
# measures layer 4, so the broader claim outruns the data. Verify: if the sweep
# in the exploratory section covers other layers, this can widen again.
```

An HTML comment inside a Markdown string works when the note has to sit beside one specific paragraph; it stays invisible in the render. Make it visible only when a reader of the published report benefits from it. The marker is greppable either way, so a review pass can find every prior decision before touching the same text.

**A note should only record the change and its warrant:** what the report now claims, and why that follows from the data. It is the same category of thing as a code comment explaining a non-obvious invariant, which is why the next round may read it. It never carries a judgment of the report's quality, a round's confidence, or anything phrased as "I suspect" or "this felt weak", since that primes the next reader instead of informing them. Observations of that kind go in the round's own report, under `Tensions`, where they reach the supervisor and stop.

**A note you would reverse is a finding.** Wanting to undo a recorded decision usually means the claim is doing two jobs at once, and each reviewer is right about a different one. Say so, name both readings, and stop. The resolution is structural (split the hypothesis into two tracks, drop one, or state the scope that separates them) and it needs the human, because it changes what the experiment claims.

Commissioning the rounds — which pass to run when, how to brief a reviewer, when to stop and escalate, and the lighter pass for prose alone — is in [references/review-passes.md](references/review-passes.md). Always read this if you are the lead.

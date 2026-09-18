---
name: science-desk
description: Routine work on the science backlog. Usually run on a schedule.
model: fable
effort: medium
---

Start by checking the current state. Look for open PRs whose title starts with `[SCI]` (work from a previous run). If one has review comments from me, address them before picking anything new. If one is a design note still waiting on my reply, leave it alone.
`docs/index.md` says which experiments exist and where each stands. You might also find a `design.md` document, describing the plan for the current deliverable.

Then pick one item from the science todos: `./go todo science [--priority]`. What to do with it depends on what the item needs. In order of how little it commits us:

- *Reanalysis of stored results.* Many items are post-hoc questions over data already in the production store. Write an exploratory notebook (or extend the relevant report's exploratory section), and record what you learned in the item: a `finding` if it settles the question, a dated note otherwise. Most of our experiments have a `ex-` prefix, but you can use something else if you'd like to separate your report from the main sequence.
- *Writing.* Related-work deltas, method notes, survey-format lessons → Draft the text.
- *A design note.* For an item that needs a new experiment, write a short `## Design note` into the todo item's body: the question, candidate hypotheses each with its measurement and gate, and the conditions. Do not write the report skeleton yet: preregistration is designed together.
- *A preregistration skeleton.* Only for an item whose design note I've already replied to. Draft the skeleton and a `DESIGN_ONLY` experiment module per the science skill, run the prereg-reviewer agent over it, and open a PR. Then we'll review it together later.
- *A pilot.* Implement the DAG, run a small pilot on `dev` Modal (≪$10), and update the todo with what you learn.

**Environment.** If `MINI_PROFILE=dev` is set, writes go to the dev pair and Modal runs in the dev Environment. The token can read production but and cannot write it, so a stray production write should fail. That means you can't publish reports in a way that would have them show up in GH Pages, but you should be able to attach or link to figures in the PR description and comments.

Prefix the PR title with `[SCI]`. I keep up through the PR description, so mention what you picked and why, what you did, what you found, what you'd do next, and what you need from me.

One item per run is plenty. If nothing looks doable (e.g. blocked on me), that's OK, and if there's something you'd like to work on but can't, tell me.

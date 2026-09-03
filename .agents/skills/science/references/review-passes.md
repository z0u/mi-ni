# Running review passes

Read this when you are the one *commissioning* a review — driving the report yourself, or running the `report-review` skill. A reviewer agent doesn't need any of it; what a reviewer needs is the `REVIEW` note convention in [SKILL.md](../SKILL.md).

## Which pass

The `report-review` skill runs the report past fresh readers who haven't seen the conversation that produced it. Use it twice, for two different questions:

- Before freezing the hypotheses, and again before running: the `prereg-reviewer` asks whether the design is sound and worth running as specified, and comes back with a run/freeze/discuss recommendation.
- Once the results are in and the report is filled, before publishing: the `results-reviewer` asks whether the results support the claims and whether a fresh reader can follow the report, with every hypothesis scored against its frozen threshold, post-hoc readings kept out of the primary sections, and figures and tables captioned and legible.

A part-filled report gets the `results-reviewer` over the sections that are done. Say which sections are in scope, so a `TODO` in a section whose turn hasn't come isn't read as an omission.

## Why rounds oscillate

Each reviewer sees the report and nothing else. That isolation is what makes them useful, and it is also why two rounds can each be locally right about a claim and correct it in opposite directions. Two mechanisms keep that in check, and they are deliberately separate.

The `REVIEW` note lives in the report. It carries a decision and its warrant forward to the next round, which is allowed to read it: the note is part of the artifact, so passing it on leaks nothing. This is what stops round 3 re-litigating a sentence round 1 settled.

The `Tensions` field lives in the reviewer's structured report. It carries a round's own reading, its doubts, its confidence, what it nearly changed, to you and no further. Nothing downstream sees it, which is what lets the next round's judgment stay independent.

Keeping those channels distinct is the whole design. A note that starts editorializing about the report's quality has drifted into the second channel's job and has begun priming the next reader; see the note-content rule in SKILL.md.

## When to stop

Escalate rather than run another round when a reviewer reverses a decision an earlier round recorded, or reports a claim pulling two ways. Another round will just flip it back. Put both readings to the human and say what would separate them: splitting the hypothesis into two tracks, dropping one, or narrowing its scope. That call changes what the experiment claims, so it is theirs.

Also stop and escalate for anything structural a reviewer couldn't fix itself, and for a disagreement about how a result should be read.

## The prose passes

Prose gets two passes regardless of whether a review round runs — see "Collaborating on a report" in SKILL.md for when they apply. No reviewer agent is involved. In one turn:

1. Write
2. Stage changes (unless you have another way to see what the agents change)
3. Hand the file and line range to the `prose-simplifier` agent
4. Run the `report-restructure` skill over the same range
5. Review both sets of changes; check for correctness

That order, and only that order. The simplifier unstacks dense sentences, splits appositives that smuggle in a definition, and surfaces buried verbs. The restructure pass then groups the result into short paragraphs that open on their findings, and cuts what the document already says elsewhere. Reversed, the simplifier would unpick the grouping.

Both edit `report.py` directly, so template expressions stay live and there is nothing to port back. Run the check afterwards:

```bash
.agents/skills/report-restructure/scripts/check-templates docs/pipeline/report.py
```

It parses the file, which catches a dropped brace, a stray brace, or an undoubled LaTeX brace, and lists any expression that went missing.

One check stays yours, because no script finds it: read the diff for a qualifier, a hedge, or a modal that went quiet, and for a referring expression whose antecedent left with a deleted sentence. Both agents are told the text is correct, so neither will notice that "would start to matter" became "matters", or that "that decay" no longer has a decay to point at.

Do not take either agent's summary of its own work. Agents doing this consistently misjudge how much they changed and overstate what they preserved. `git diff --stat` and the template check settle both in a second.

## What each pass is worth

The prose passes recover roughly 2 to 20% of a section, and most of their value is in shape rather than length: a section that is 40% figure captions, alt text, and tables has little that can move, since all three are protected.

Real length comes off at the structural level: duplication across sections, front matter that runs before the first result, a discussion that re-derives results the findings section already gave. That is several times larger, and it is only visible in the assembled document. The `report-structure` agent does that pass at the freeze and publish gates, reading the render rather than the source, since a cell that fails to render is invisible in the source and we have had one go missing.

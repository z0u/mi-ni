---
name: report-restructure
description: |
  Reshapes report prose so it can be skimmed, and cuts what repeats. Edits the
  notebook source, so template expressions survive. Does not check numbers or
  claims. Verify correctness yourself afterwards.
argument-hint: <path to report.py> [section or line range]
context: fork
model: sonnet
effort: low
tools: Read, Edit, Bash
---

We are reshaping prose so it can be read quickly. We are not checking numbers or claims.

## Workflow

1. Stage the file first, unless you have another way to see what changed.
2. Find the line range for the section. Give the `report-restructure` agent the path and that range, and nothing else — no experiment context, no conversation history. It works better without them, and its judgement about what is load-bearing has to come from the page rather than from you.
3. When it returns, run the check:

   ```bash
   .agents/skills/report-restructure/scripts/check-templates docs/pipeline/report.py
   ```

   It parses the file, which catches a dropped brace, a stray brace, or an undoubled LaTeX brace, and it lists every template expression that went missing. A lost expression is not automatically wrong, since deleting a sentence takes its expressions with it, but each one is a place to confirm the value was removed rather than frozen.
4. Read the diff for correctness. Two things the agent cannot catch:
   - A qualifier or a hedge that went quiet. It is told the text is correct, so it will not notice that "would start to matter" became "matters".
   - A referring expression whose antecedent left with a deleted sentence.
5. Measure, rather than reading the agent's summary. Agents doing this work consistently misjudge how much they changed, in both directions, and consistently overstate what they preserved. `git diff --stat` and the template check take a second each.
6. Report the flavor of the changes, not the details; those are in the diff.

## What this pass is worth

Roughly 2 to 20% of a section's words, and most of the gain is in shape rather than length. How much depends on what the section is made of: a section that is 40% figure captions, alt text, and tables has little that can move, because all three are protected. A section of unbroken prose has more.

If a report needs to lose real weight, this is the wrong pass. Duplication across sections and front matter that runs before the first result are worth several times more, and they are only visible in the assembled document — that is the `report-structure` agent's job.

## Where the register came from

The house register for these sections is lab notes, written up: short paragraphs opening on their findings, colons carrying figure keys, lists kept inline. The agent's spec describes it, along with the rule that decides how far to compress: against the immediate context, never against the sentence being deleted.

That rule was worked out by running several designs over one section whose failure modes were known, so it is a description of what held up rather than a preference. The short version: a trim is safe when everything needed to understand what remains is still on the page nearby, and unsafe when recovering the meaning needs the sentence that just left.

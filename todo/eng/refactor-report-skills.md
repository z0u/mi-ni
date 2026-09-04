---
status: open
tags: [reports, skills, agents, methodology]
opened: 2026-09-02
bundle: skills-workflow
---

# Refactor the report-writing workflow

Currently, we have several skills and agent configs detailing different parts of the report-writing workflow, including several phases of review. It has become almost inscrutable.

Suggestion:

1. Map out the current state to understand what the full workflow is.
2. Redesign it, taking all steps into account and carefully considering roles and responsibilities, and which models are best suited to each part.
3. Re-write it, still as a collection of skills and agent configs, but maybe in a plugin — so they're all in one place and clearly related to each other. Follow the `style-md` and `style-skills`.
4. Document it, including a Mermaid flowchart, to make it more scrutable.
5. Ask for fresh-eyes reviews for both correctness and style (separately) with clean context.

These skills are the backbone of our work and they're worth getting right.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/refactor-report-skills.md`](https://github.com/z0u/sca2/blob/main/todo/eng/refactor-report-skills.md) there) with the code it describes.

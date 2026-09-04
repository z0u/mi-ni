---
status: open
tags: [agents, skills]
bundle: skills-workflow
---
# Skills/agents reorg

(see the model-routing section of `CLAUDE.md` for context):

- Add a routing table for writing-adjacent tooling — one front door (`report-review` for reports; a short "which tool when" list for the rest) so the entry point is discoverable on the human's turn.
- The `.agents/skills` directory mixes three kinds of thing: conventions (writing, style-*, alt-text, science), runbooks that fork and drive agents (report-review, report-restructure), and references (report-render, mi-ni). Consider naming or a one-line "kind:" tag in each SKILL.md to make the taxonomy visible.
- Consider extracting the transferable set (writing, style-*, alt-text, text-lint, report pipeline skills + agents) into a plugin. Projects started from this template take the skills as a copy today, and the copies drift, which is what the two-way ports between this repo and sca2 keep paying for; a plugin would let a project take a dependency instead. Keep repo-specific skills local. Decide after the routing table has settled, so the plugin ships a stable interface.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/skills-agents-reorg.md`](https://github.com/z0u/sca2/blob/main/todo/eng/skills-agents-reorg.md) there), where the plugin was framed as "for the next milestone repo"; from this side it is the template's skills that would ship as one. Sits with [refactor-report-skills](./refactor-report-skills.md) in the `skills-workflow` bundle.

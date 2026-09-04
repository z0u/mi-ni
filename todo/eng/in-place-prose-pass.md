---
status: done
tags: [agents, reports, skills]
opened: 2026-08-01
closed: 2026-08-01
---
# In-place mode for the prose pass

Kept here for the measurements, which are why the pass was redesigned rather than optimized. Seven candidate designs ran over one section whose failure modes were known, varying model, substrate and instructions. Results: told what is load-bearing, an agent finds 2 to 5% of a section to cut, not the 48% the old lint appeared to deliver — that figure came from cutting figure captions by 67%, which is content loss rather than trimming. Instructions dominate; the model barely matters (Sonnet and Opus given the same spec made identical cuts). Achievable reduction is set by how much of a section is protected material, so 40% captions and tables caps a section near 7% while unbroken prose reaches 20%. Source-native won on both counts asked of it. Markdown does not corrupt f-strings, but every edit becomes manual reconciliation that scales with the cut; and `ast.parse` validates template expressions for free, since Python compiles f-strings and a dropped or undoubled brace is a SyntaxError. Opus editing source cut slightly deeper than the same prompt on Markdown, so handling the code cost nothing. Shipped as the `report-restructure` skill and agent, with `scripts/check-templates` as the gate. `text-linter` and `writing-lint` are gone. Structural work — where the real length is, roughly five times the prose pass — went to the new `report-structure` agent.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/in-place-prose-pass.md`](https://github.com/z0u/sca2/blob/main/todo/eng/in-place-prose-pass.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

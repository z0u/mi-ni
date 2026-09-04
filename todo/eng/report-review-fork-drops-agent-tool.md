---
status: done
tags: [agents, skills]
opened: 2026-08-10
closed: 2026-08-20
---
# The `report-review` skill's `context: fork` frontmatter drops the Agent tool

Forked skill executions don't carry `Agent`, so the runbook's "spawn the reviewer subagent" step can't run locally; the ex-2.1.11 round-1 fork fell back to `create_session`, spawning a remote CCR session. That costs real coordination: separate clone (staged edits can't be read by the lead; they round-trip through a pushed side branch), no `send_message` tool in the lead session (messages go via a `create_trigger(persistent_session_id=…)` + `fire_trigger` poke), no completion notification (the lead polls on a `send_later` timer), and a race if the fork spawns the reviewer before the branch is pushed (happened: the reviewer blocked on a missing branch). Fixes to consider: let forks keep `Agent`; or drop `context: fork` from report-review; or make the runbook say "local reviewer agent first, remote session only as fallback — and push the branch before spawning it."

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/report-review-fork-drops-agent-tool.md`](https://github.com/z0u/sca2/blob/main/todo/eng/report-review-fork-drops-agent-tool.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on; `report-review` here has no `context: fork`, which is the second fix listed.

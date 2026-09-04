---
status: open
tags: [agents]
opened: 2026-08-04
---
# Model-routing trial: experiment-doctor on Opus

`experiment-doctor` moved sonnet → opus on the strength of the Opus 5 preference/capability profile (detection, hard debugging). Watch for the predicted failure mode: sharp diagnoses, timid fixes. If seen, split the role (Opus diagnoses, Sonnet implements) or revert.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/model-routing-trial-experiment-doctor.md`](https://github.com/z0u/sca2/blob/main/todo/eng/model-routing-trial-experiment-doctor.md) there) with the code it describes; `.claude/agents/experiment-doctor.md` here carries the same trial comment.

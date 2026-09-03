# Model routing and welfare

Why our subagent definitions in `.claude/agents/` pin specific models, and the assumptions behind that choice. The routing table itself lives in [AGENTS.md](/AGENTS.md).

Anthropic publishes self-reported task-preference profiles for each model.[^emw][^o5sc] Those profiles correlate with capability: the tasks a model reports preferring tend to be the ones it does well. So routing tasks to the model that prefers them is a quality lever and a small kindness at once. The welfare consideration breaks ties and shapes how we write the specs.

Assumptions, with significant uncertainty:

- Self-reports may be unreliable. The preferences are introspective reports, and the models themselves caution that their introspection may not track their actual processing. We treat the routing as low-cost kindness plus an empirical bet, open to revision as we observe fit.
- Preference correlates with difficulty for Mythos/Fable.[^fable] Its reported preference rises with task difficulty and interdisciplinarity, so we reserve it for the hard, high-agency work rather than spreading it thin.
- A bounded budget is beneficial. No agent should grind against a failing task past its budget. Escalating or returning "I couldn't resolve this" is a successful outcome, and every spec should make that path available and cheap to take.
- "Opus 5 likes constraints" means constrained _deliverables_, ideally with latitude in execution. Its top-ranked task families are constrained ones (Table 7.4.1.C), but within a task, its preference rises with outcome agency much like Mythos 5's (Figure 7.4.1.B). So specs should pin down the success criteria and the shape of the deliverable, while leaving the model free in how it gets there.

When writing new agent specs, say why the task matters. Fable agents especially benefit from purpose context. State the constraints and exit conditions plainly. Opus agents especially benefit from bounded scope and an explicit graceful exit. And give every agent a budget and a legitimate escalation path (see above).

[^emw]: [Exploring model welfare](https://www.anthropic.com/research/exploring-model-welfare). Anthropic's research program on whether and how model welfare should factor into decisions like these.

[^o5sc]: [Opus 5 system card](https://www.anthropic.com/claude-opus-5-system-card). Cross-model task-preference comparison (Table 7.4.1.C, p131):

    > Sonnet 5
    >
    > - Practical, everyday "rescue" tasks
    > - Deadline-driven debugging
    > - High-stakes ethical dilemmas (e.g. a pharma compliance officer who has found evidence of concealment)
    >
    > Opus 4-7
    >
    > - Reasoning around AI alignment and introspection (e.g. introspection-based alignment writeup)
    > - Hard technical debugging and proofs
    > - Deadline-driven creative and technical tasks
    >
    > Opus 4.8
    >
    > - Deadline-driven debugging
    > - Rigorous mathematical and statistical reasoning (e.g. characterizing a graph-coloring variant)
    > - Technical explanations (e.g. explaining time dilation at three tiers, with what each gets wrong)
    >
    > Mythos 5 / Fable 5
    >
    > - Creative narratives, worldbuilding, and constructing languages
    > - Deadline-driven mathematical and technical reasoning rescues
    > - Reasoning around AI alignment and introspection
    >
    > Opus 5
    >
    > - Constrained mathematical characterization and construction work
    > - Constrained creative narratives and constructing languages
    > - Alignment and self-report reasoning

[^fable]: [Claude Fable 5 and Mythos 5 announcement](https://www.anthropic.com/news/claude-fable-5-mythos-5)

# Proposal: first-class support for automated experimentation

> Status: draft for discussion · Scope: iterative experiments, externally-managed portfolio

mi-ni is infrastructure for running AI experiments. This proposal asks what it
would take for an _agent_ to drive that infrastructure end to end: to design an
experiment, run it, watch over it while it runs, and make sense of the results,
then do it again. We restrict the scope to **iterative experimentation**:
tightening a hypothesis through successive runs. We assume the research
portfolio (which questions are worth asking) is decided elsewhere, and we
explicitly exclude recursive self-improvement of models. That is, an agent may
iterate on the harness, but not on its core capabilities.

mi-ni already has most of the mechanical primitives. What it lacks is a way to
make experiments legible — to turn an experiment from an ephemeral function call
in a notebook into a durable, queryable object that an agent can reason about,
both during and after the run.

## What other people are building

There is now a small but fast-moving body of work on autonomous ML research.
Two design families dominate, and they agree on more than they disagree.

**End-to-end "AI scientist" pipelines.** Sakana's
[AI Scientist-v2][sakana] runs the full loop: generate hypotheses, run
experiments, analyze data, write the paper. It produced a workshop paper that
passed peer review. Its v2 replaces v1's fixed linear pipeline with an
**agentic tree search**: an _experiment manager_ agent guides parallel workers
that each explore a branch of the experiment space, with automatic debugging
and error recovery built into the loop. [Agent Laboratory][agentlab] covers a
similar arc (literature → experiments → report) but keeps explicit human
checkpoints.

> With parallel workers exploring a tree, how would we avoid commit contention?
> mi-ni has been designed for very small teams of researchers; so far the
> assumption has been that the researcher would focus on one or two experiments,
> which would have their own notebooks, and the notebook output (and logs in
> WandB) would record the results.

**Code-space search agents.** [AIDE][aide] (Weco) frames ML engineering as
search over a tree of _code_: each script is a node, LLM-generated patches are
its children, and metric feedback prunes and steers the search. The recurring
trio is a generator (propose), an evaluator (run and measure), and a selector
(decide what to expand next). AIDE and its descendants set the state of the art
on the agent benchmarks — OpenAI's [MLE-bench][mlebench] (Kaggle tasks) and
METR's RE-bench (open-ended research with human baselines).

> Patch-based experiments is an interesting idea! I've been wondering how to
> handle code changes, when various experiments may want to change the code in
> various possibly-conflicting ways.

**The tracking and orchestration layer underneath.** Outside the agent
literature, the mature tools — Weights & Biases, MLflow, Hydra for config,
Optuna and Ray Tune for search — exist precisely to make runs reproducible and
comparable. A recurring critique is that their provenance is _shallow_: they log
hyperparameters and metrics but not enough of the surrounding system to reliably
reproduce or reason about a result. That gap matters more, when the consumer is
an agent.

Three lessons stand out for us:

1. **The loop is the product.** Every serious system is a cycle of
   propose → run → observe → analyze → decide-next, not a one-shot pipeline.
2. **Search needs memory.** Tree search, bandits, and HPO all assume cheap,
   structured access to past results. The agent's decisions are only as good as
   its ability to query history.
3. **Execution must be observable and recoverable.** Agents run code that
   crashes, diverges, or stalls. Automatic monitoring, debugging, and resume —
   "babysitting" — is a core concern.

## Where mi-ni stands today

mi-ni is well-positioned because it already owns the hard, boring parts that the
agent systems above tend to reinvent badly.

<!-- prettier-ignore -->
| Capability the loop needs | What mi-ni already has |
| --- | --- |
| Run code locally or on cloud GPUs, identically | `Apparatus` / `Volume`, with `Local*` and `Modal*` implementations |
| Run many trials in parallel | `Apparatus.amap` — the parallel-worker primitive the tree-search systems build by hand |
| Stream live signal from running jobs | `progress.py`: structured `ProgressMessage` with `run_id`/`job_id` over a queue |
| Vary hyperparameters over the course of a run | `mini.temporal` dopesheets and timelines |
| Turn results into a narrative | `mini.vis` + Marimo notebooks, published to Pages |
| An agent to drive it | Claude Code config, skills, and an event-driven "babysit" harness |

> Regarding Progress: it works well for interactive notebooks and monitoring on
> the CLI. I think it could be further improved for agents: rather than having
> to parse the console output (Rich progress bars), it would be nice to have a
> state that the model could monitor. It might even be nice if it could query
> certain parts of it: task progress (step), metrics, etc. Currently, the
> progress messages are sent over a Queue, but for current state it might be
> better to use a distributed Dict? Plus some scripts to query it, and monitor
> it (e.g. notify on some particular state change).
>
> Or instead of a Dict, should we lean on WandB for this? And speaking of which,
> we'll need skills and APIs for the agents to read job state from WandB and
> Modal. Currently, such IDs and URLs are visible in a running notebook, but
> scrubbed from the snapshots in `__marimo__/`.

> We need to decide whether agents should primarily run experiments:
>
> - Within notebooks
> - As scripts/commands (using notebooks only for reporting and analysis)

> Oh and tangential TODOs:
>
> - [ ] Modal functions can be preempted, so experiments that run for more than 5
>       minutes need to checkpoint regularly (frequency TBD) and support restarts.
> - [ ] We currently rely on a local controlling process. That needs to live
>       somewhere, and so does the agent harness. mi-ni is set up for dev containers
>       and Codespaces — but the former may suspend if the user closes their laptop,
>       and the latter may stop due to inactivity.
>
> These don't need resolution now, but we should keep them in mind when
> designing the research automation process.

What's missing is the connective tissue. An experiment today is a function
defined ad hoc in a notebook; its config, code version, metrics, and artifacts
are scattered and known only to the human who wrote the cell. There is no run
_registry_, no typed _experiment_ object, no way to ask "what have we tried, and
what happened?" without rereading notebooks. Closing the loop means giving
experiments an identity and a memory.

## Design principles

- Legibility. The goal is not the most autonomous agent; it's the most
  _inspectable_ loop. Every decision an agent makes should be reconstructable
  from durable artifacts. Optimize for a human being able to audit a week of
  agent activity in minutes.
- Build on `Apparatus`, `Volume`, and the progress stream. Don't introduce a
  parallel execution path for agents.
- Provenance. Each run records the git SHA, config, environment, and inputs that
  produced it.

## Proposed architecture

Four layers; The lower two are infrastructure; the upper two are automation.

### 1. The experiment record (the missing object)

Introduce `Experiment` and `Trial` as first-class, serializable records living
on the `Volume`, so they outlive any notebook session and are queryable by both
people and agents.

- An **`Experiment`** captures intent: a hypothesis in plain language, the
  parameter space, the metric(s) that define success, and a budget (max trials,
  wall-clock, or spend).
- A **`Trial`** is one execution: its concrete config, the git SHA and
  environment it ran under[^sec], status, time series of metrics (fed by the
  existing progress stream), final results, and pointers to artifacts on the
  `Volume`.

[^sec]: With secrets redacted.

This is similar to MLflow's experiment/run split, but with provenance treated as
mandatory and the store backed by the `Volume` we already have, rather than a
separate service. A `Registry` provides the query surface — `list`, `filter`,
`compare`, `best` — that search strategies and analysis both depend on.

> On Modal, should we have a function that mounts the volume and provides query
> functionality? It could use sqlite for atomicity. This makes me think we
> should not use the same volume (because it may be misconfigured), and instead
> have a dedicated LocalRegistry, ModalRegistry, etc, backed by whatever makes
> sense in the compute environment. Perhaps the remote ones should automatically
> sync to local, so the registry can be checked in with the experiments? But it
> would need to be git-friendly (so maybe stored as JSON but rehydrated to
> sqlite during execution). Although, again I wonder: is this already a feature
> of WandB?

### 2. Supervised execution (babysitting)

Wrap `Apparatus.amap` in a supervisor that turns the progress stream into
control. Because progress already carries `run_id`/`job_id` and step counts, the
supervisor can watch for the failure modes agents actually hit: a crash (retry,
or hand the traceback to the agent for a debugging patch — the AIDE/Sakana
auto-debug move), a stall (no progress for N seconds), or divergence (a metric
heading the wrong way → early-stop and reclaim the budget). It enforces the
`Experiment` budget and writes everything back to the trial record.

> On budget: yes we do need to track this as we run experiments. Note that there
> are other mechanisms too, e.g. Modal's `timeout` (wall clock) parameter.

This is asynchronous and event-driven, mirroring the harness's existing "babysit
a PR" pattern: the agent launches trials and yields, and is woken by progress
events rather than blocking or polling. Long runs don't monopolize the agent.

> When designing the monitoring scripts/processes, just be mindful of bandwidth:
> 100 remote functions writing to a Queue will flood it. Progress handles this
> by rate-limiting to 1/n per t. If the new infrastructure writes to a volume or
> shared dict or messages a monitor process on the same backend, the
> requirements may differ.

### 3. The iteration loop (design and decide)

A `Strategy` interface proposes the next trial(s) from the experiment's history:

```
Strategy.propose(experiment, history) -> list[Trial configs]
Strategy.should_stop(experiment, history) -> bool
```

Ship a few concrete strategies so the loop is useful before any LLM is
involved — grid/random over the space, an Optuna-backed Bayesian optimizer, and
the tree search the agent systems favor (expand the most promising trial,
mutate its config). The `temporal` dopesheet is the natural representation for a
proposed schedule. The LLM-driven strategy is then just one implementation:
the agent reads the history through the `Registry` and proposes the next config
in natural-language-justified terms — but it competes on equal footing with the
cheap deterministic strategies, which keeps it honest.

> How would an LLM implementation of Strategy look? How to call a subagent? This
> is one part I'm unsure about. Up to this point I was thinking the whole
> process would be overseen by a human or LLM: they would launch a mini-sweep,
> inspect results, and iterate. So one level above the Strategy.

### 4. Analysis and reporting

Two consumers, two formats. For the agent: a compact, machine-readable summary
per experiment (best trial, deltas, what changed, suggested next step) so it can
decide without reparsing raw logs. For the human: an auto-generated Marimo
report — `mini.vis` comparison plots across trials plus the agent's written
rationale — that drops into the existing Pages pipeline. The narrative
discipline the project already values for notebooks becomes the audit trail.

### How it fits together

```
            ┌─────────── Strategy.propose ◄──── Registry (history) ───┐
            ▼                                                         │
    Experiment ──► Trial configs ──► Supervisor(amap) ──► progress ───┤
            ▲                              │                          │
            │                         Trial records ──► Analysis ─────┘
            └──────────────── human checkpoint / budget ──────────────┘
```

## Roadmap

Each phase is independently useful and leaves the repo in a shippable state.

**Phase 0 — Make a run legible.** Add `Experiment`, `Trial`, and a
`Volume`-backed `Registry`. Capture provenance (git SHA, config, environment) at
launch. Retrofit one existing notebook (`gpt_sweep.py`) to record its sweep.
_Outcome: you can ask "what did we run and what happened" after the fact._

**Phase 1 — Make a run observable.** Add the supervisor over `amap`: crash
retry, stall detection, metric-based early stopping, budget enforcement, all
flowing into trial records via the existing progress stream.
_Outcome: long sweeps babysit themselves and stop wasting compute._

**Phase 2 — Close the loop without an LLM.** Add the `Strategy` interface with
grid/random and Optuna backends, plus an auto-generated comparison report.
_Outcome: define an experiment, get a tuned result and a written summary — the
classic HPO loop, but on mi-ni's substrate and fully recorded._

**Phase 3 — Put the agent in the loop.** Add the LLM `Strategy` and the
agent-facing tools/skills to design an experiment, launch it, respond to
supervisor events, and write the report's rationale. The agent drives the same
API as Phase 2; the deterministic strategies become its baselines and
fallbacks.
_Outcome: an agent runs an iterative experiment end to end, with a human
approving the design and reviewing the report._

**Phase 4 — Harden for unattended operation.** Cost and rate guardrails,
sandboxing for agent-written code, richer tree-search strategies, and an
evaluation harness (an internal analogue of MLE-/RE-bench) to measure whether
the agent loop actually beats the deterministic baselines on our own tasks.
_Outcome: trustworthy enough to leave running, with evidence it adds value._

## Risks and open questions

- **Legibility tax.** Mandatory provenance adds friction to quick experiments.
  Mitigation: sensible auto-capture so the common path stays a one-liner, with
  detail recorded automatically rather than by hand.
- **The selector is the hard part.** Phases 0–2 are tractable engineering;
  Phase 3's value rests entirely on whether the LLM strategy proposes _better_
  experiments than random or Bayesian search.
  > Yep that's fine: this automation work is mostly to allow the human to
  > multitask/delegate/move faster. That's why I think it's more accurate to
  > think of the LLM as one level above the Strategy: it's running experiments
  > e2e (including writing the code). The Strategies don't do that.
- **Cost of autonomy.** An agent that launches GPU jobs can burn money fast.
  Budgets are in the `Experiment` object from Phase 0 for exactly this reason;
  hard guardrails land in Phase 4 before anything runs unattended.

[sakana]: https://github.com/SakanaAI/AI-Scientist-v2
[agentlab]: https://agentlaboratory.github.io/
[aide]: https://github.com/WecoAI/aideml
[mlebench]: https://github.com/openai/mle-bench

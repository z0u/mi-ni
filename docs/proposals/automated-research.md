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

**Code-space search agents.** [AIDE][aide] (Weco) frames ML engineering as
search over a tree of _code_: each script is a node, LLM-generated patches are
its children, and metric feedback prunes and steers the search. The recurring
trio is a generator (propose), an evaluator (run and measure), and a selector
(decide what to expand next). AIDE and its descendants set the state of the art
on the agent benchmarks — OpenAI's [MLE-bench][mlebench] (Kaggle tasks) and
METR's RE-bench (open-ended research with human baselines).

**The tracking and orchestration layer underneath.** Outside the agent
literature, the mature tools — Weights & Biases, MLflow, Hydra for config,
Optuna and Ray Tune for search — exist precisely to make runs reproducible and
comparable. A recurring critique is that their provenance is _shallow_: they log
hyperparameters and metrics but not enough of the surrounding system to reliably
reproduce or reason about a result. That gap matters more when the consumer is
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

Two decisions about how agents use this substrate shape everything below.

**Agents run experiments as scripts; notebooks are for reporting.** A headless
script is parametrizable, observable, and survives being killed and restarted —
all properties an agent needs and a notebook lacks. Notebooks remain the place
where results become a narrative for humans. This also sidesteps a real problem:
the run IDs and URLs an agent needs (WandB, Modal) are visible in a live
notebook but scrubbed from the `__marimo__/` snapshots, so they can't be the
durable record. The trial record is.

**The progress stream stays the transport; current state lives in the registry.**
The `Queue` is the right primitive for streaming and rate-limiting, but an agent
wants to _query_ state (latest step, metrics, status), not parse a Rich progress
bar. So the supervisor folds the stream into the trial record, which becomes the
queryable live state — no separate distributed dict. Metrics time series are
WandB's job; mi-ni reads from it rather than reimplementing it (see below).

Two infrastructure constraints to keep in mind throughout, even though neither
needs solving now: Modal preempts long functions, so any trial over a few
minutes must checkpoint and resume (Phase 1); and the controlling process needs
a durable home — a dev container suspends with the laptop, a Codespace stops on
inactivity — which argues for a controller that holds no state of its own and
can resume from the registry (see "Where the controller lives").

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

Four layers. The lower two are infrastructure; the upper two, automation.

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
mandatory. A `Registry` provides the query surface — `list`, `filter`,
`compare`, `best` — that search strategies and analysis both depend on.

The storage design follows the `Apparatus`/`Volume` pattern: a `Registry`
abstraction with `Local` and `Modal` implementations, each backed by whatever
suits its environment, and deliberately _not_ the same volume that holds
experiment data (too easy to misconfigure into corruption). The canonical store
is git-friendly JSON — one file per experiment and trial, checked in alongside
the code so history travels with the repo — rehydrated into sqlite for fast
querying during a run. JSON is the source of truth; sqlite is a disposable
index. On Modal, a function mounts the registry backend and serves queries;
remote state syncs back to local so the checked-in record stays authoritative.

This is intentionally complementary to WandB, not a replacement. WandB is the
metrics and observability plane — live curves, time series, the dashboard a
human watches. The registry is the control plane — what was run, under which
SHA, with what budget, and where the artifacts are. The registry stores the
WandB and Modal IDs so an agent can follow them; reading metrics back is a thin
skill over the WandB and Modal APIs (Phase 0). A single writer keeps this
honest: trial records are written by the supervisor, never by the trials
themselves, which avoids both sqlite write contention and the fan-in flood that
would come from many workers writing at once.

### 2. Supervised execution (babysitting)

Wrap `Apparatus.amap` in a supervisor that turns the progress stream into
control. Because progress already carries `run_id`/`job_id` and step counts, the
supervisor can watch for the failure modes agents actually hit: a crash (retry,
or hand the traceback to the agent for a debugging patch — the AIDE/Sakana
auto-debug move), a stall (no progress for N seconds), or divergence (a metric
heading the wrong way → early-stop and reclaim the budget). It also handles
Modal preemption: because trials checkpoint to the `Volume`, a preempted or
restarted trial resumes from its last checkpoint rather than from zero. The
supervisor writes everything back to the trial record.

Budget enforcement is layered. The `Experiment` budget (max trials, spend) is
the coarse ceiling the supervisor watches; per-trial wall-clock is delegated to
Modal's `timeout` parameter, which kills a runaway trial at the source rather
than waiting for the supervisor to notice.

This is asynchronous and event-driven, mirroring the harness's existing "babysit
a PR" pattern: the agent launches trials and yields, and is woken by progress
events rather than blocking or polling. Long runs don't monopolize the agent.
Fan-in is the thing to watch — a hundred workers writing to one queue will flood
it. The progress stream already rate-limits to one message per worker per
interval; the supervisor is the single consumer and batches its writes to the
registry, so the back end never sees a hundred concurrent writers.

### 3. The iteration loop (design and decide)

There are two nested loops here, and keeping them distinct is the key design
decision. The **inner loop** searches a _fixed_ experiment — same code, varying
configuration. The **outer loop** is the researcher's: changing the code,
defining new experiments, interpreting results, deciding what to ask next. An
LLM agent operates the outer loop; the inner loop is cheap, deterministic, and
needs no LLM at all.

The inner loop is a `Strategy` — a config proposer over a fixed experiment:

```
Strategy.propose(experiment, history) -> list[Trial configs]
Strategy.should_stop(experiment, history) -> bool
```

Ship a few concrete strategies — grid/random over the space, an Optuna-backed
Bayesian optimizer, and a config-space tree search (expand the most promising
trial, mutate its config). The `temporal` dopesheet is the natural
representation for a proposed schedule. None of these touch code; they only move
within the parameter space the experiment declares.

The outer loop is not another `Strategy` — it's the agent harness itself
(Claude Code, or a subagent it spawns) doing what a researcher does: reading the
registry, editing code, defining an experiment, launching a sweep (which may
delegate to a `Strategy`), reading the report, and iterating. This is why
agent-written _code_ variation lives here and not in the inner loop, and it's
the right altitude for the human checkpoint: a person reviews an experiment's
design and its report, not every config the inner loop proposes. The point of
automating this loop is to let the researcher delegate and multitask, not to
replace their judgement about what is worth trying.

### Config-space vs code-space variation

The two loops also resolve how to vary code without contention — the thing the
tree-search systems handle by mutating a shared working tree. We don't. Two
kinds of variation, kept separate:

- **Config-space** variation is the inner loop: one immutable code version (a
  git SHA), many configs, run in parallel via `amap`. No code changes, so no
  conflict — this is the common case and the default.
- **Code-space** variation is the outer loop: the agent proposes a code change
  as a branch or worktree over a base SHA, never by editing the shared tree
  in place. Each trial records the SHA (and diff) it ran under, so conflicting
  variants coexist as branches and a human merges the winner. This fits mi-ni's
  small-team, one-or-two-experiments-per-researcher model rather than fighting
  it.

### 4. Analysis and reporting

Two consumers, two formats. For the agent: a compact, machine-readable summary
per experiment (best trial, deltas, what changed, suggested next step) so it can
decide without reparsing raw logs. For the human: an auto-generated Marimo
report — `mini.vis` comparison plots across trials plus the agent's written
rationale — that drops into the existing Pages pipeline. The narrative
discipline the project already values for notebooks becomes the audit trail.

### How it fits together

```
  OUTER LOOP (agent / researcher)
  edit code · define experiment · read report · decide next ── human checkpoint
        │                                              ▲
        ▼                                              │
  ┌─ Experiment ──────────────────────────────── Analysis ─┐   ← report (Marimo)
  │     │                                            ▲      │
  │     ▼     INNER LOOP (Strategy)                  │      │
  │  Strategy.propose ──► Trial configs ──► Supervisor(amap)│
  │     ▲                                       │           │
  │     └────────── Registry (history) ◄── Trial records ◄──┘
  └─────────────────────────────────────────────── budget ─┘
```

### Where the controller lives

Both loops need a process to drive them, and that process can't assume a human's
laptop stays awake. Because the registry is the source of truth and the
supervisor batches its state there, the controller can hold nothing of its own —
any controller can resume an experiment from the registry. That makes the
deployment question answerable rather than blocking: run the controller wherever
is durable (a Modal app or scheduled function is the obvious candidate, since
the compute already lives there), and treat the dev container or Codespace as a
place to launch and inspect, not the thing the loop depends on staying alive.
This needs design work, but nothing in the architecture above forecloses it.

## Roadmap

Each phase is independently useful and leaves the repo in a shippable state.

**Phase 0 — Make a run legible.** Add `Experiment`, `Trial`, and a
`Volume`-backed `Registry`. Capture provenance (git SHA, config, environment) at
launch. Retrofit one existing notebook (`gpt_sweep.py`) to record its sweep.
_Outcome: you can ask "what did we run and what happened" after the fact._

**Phase 1 — Make a run observable.** Add the supervisor over `amap`: crash
retry, stall detection, metric-based early stopping, budget enforcement, and
checkpoint/resume so trials survive Modal preemption — all flowing into trial
records via the existing progress stream.
_Outcome: long sweeps babysit themselves and stop wasting compute._

**Phase 2 — Close the loop without an LLM.** Add the `Strategy` interface with
grid/random and Optuna backends, plus an auto-generated comparison report.
_Outcome: define an experiment, get a tuned result and a written summary — the
classic HPO loop, but on mi-ni's substrate and fully recorded._

**Phase 3 — Put the agent in the outer loop.** Give the agent harness the
tools and skills to do what a researcher does: read the registry, edit code,
define an experiment, launch a sweep (delegating the inner search to a Phase 2
`Strategy`), respond to supervisor events, and write the report's rationale.
The agent doesn't replace the strategies — it _uses_ them, and authors the code
they search over.
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
- **The value is throughput, not a better optimizer.** The deterministic
  strategies will out-search an LLM at pure hyperparameter tuning, and that's
  fine — they're the inner loop. The agent earns its keep at the outer loop, by
  running experiments end to end (including writing the code) so the researcher
  can delegate and multitask. Phases 0–2 are tractable engineering; the open
  question is whether the agent's _end-to-end_ throughput is good enough to
  trust, which is what the Phase 4 eval harness measures.
- **Cost of autonomy.** An agent that launches GPU jobs can burn money fast.
  Budgets are in the `Experiment` object from Phase 0 for exactly this reason;
  hard guardrails land in Phase 4 before anything runs unattended.
- **The controller has no home yet.** The "where the controller lives" question
  is real and unsolved. The architecture keeps it answerable by holding all
  state in the registry, but a durable, restartable controller still needs to be
  designed before anything runs unattended.

[sakana]: https://github.com/SakanaAI/AI-Scientist-v2
[agentlab]: https://agentlaboratory.github.io/
[aide]: https://github.com/WecoAI/aideml
[mlebench]: https://github.com/openai/mle-bench

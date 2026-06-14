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
make experiments legible: to turn an experiment from an ephemeral function call
in a notebook into a durable, queryable object an agent can reason about, both
during and after the run.

## What other people are building

There is now a small but fast-moving body of work on autonomous ML research.
Two design families dominate, and they agree on more than they disagree.

**End-to-end "AI scientist" pipelines.** Sakana's
[AI Scientist-v2][sakana] runs the full loop: generate hypotheses, run
experiments, analyze data, write the paper. It produced a workshop paper that
passed peer review. Its v2 replaces v1's fixed linear pipeline with an
_agentic tree search_: an _experiment manager_ agent guides parallel workers
that each explore a branch of the experiment space, with automatic debugging
and error recovery built into the loop. [Agent Laboratory][agentlab] covers a
similar arc (literature → experiments → report) but keeps explicit human
checkpoints.

**Code-space search agents.** [AIDE][aide] (Weco) frames ML engineering as
search over a tree of _code_: each script is a node, LLM-generated patches are
its children, and metric feedback prunes and steers the search. The recurring
trio is a generator (propose), an evaluator (run and measure), and a selector
(decide what to expand next). AIDE and its descendants set the state of the art
on the agent benchmarks: OpenAI's [MLE-bench][mlebench] (Kaggle tasks) and
METR's RE-bench (open-ended research with human baselines).

**The tracking and orchestration layer underneath.** Outside the agent
literature, the mature tools (Weights & Biases, MLflow, Hydra for config, Optuna
and Ray Tune for search) exist precisely to make runs reproducible and
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
   crashes, diverges, or stalls. Automatic monitoring, debugging, and resume
   ("babysitting") is a core concern.

## Where mi-ni stands today

mi-ni is well-positioned because it already owns the hard, boring parts that the
agent systems above tend to reinvent badly.

<!-- prettier-ignore -->
| Capability the loop needs | What mi-ni already has |
| --- | --- |
| Run code locally or on cloud GPUs, identically | `Apparatus` / `Volume`, with `Local*` and `Modal*` implementations |
| Run many trials in parallel | `Apparatus.amap`: the parallel-worker primitive the tree-search systems build by hand |
| Stream live signal from running jobs | `progress.py`: structured `ProgressMessage` with `run_id`/`job_id` over a queue |
| Vary hyperparameters over the course of a run | `mini.temporal` dopesheets and timelines |
| Turn results into a narrative | `mini.vis` + Marimo notebooks, published to Pages |
| An agent to drive it | Claude Code config, skills, and an event-driven "babysit" harness |

Two decisions about how agents use this substrate shape everything below.

**Agents run experiments as scripts; notebooks are for reporting.** A headless
script is parametrizable, observable, and survives being killed and restarted,
all properties an agent needs and a notebook lacks. Notebooks remain the place
where results become a narrative for humans. This also sidesteps a real problem:
the run IDs and URLs an agent needs (WandB, Modal) are visible in a live
notebook but scrubbed from the `__marimo__/` snapshots, so they can't be the
durable record. The trial record is.

**The progress stream stays the transport; current state lives in the registry.**
The `Queue` is the right primitive for streaming and rate-limiting, but an agent
wants to _query_ state (latest step, metrics, status), not parse a Rich progress
bar. So the supervisor folds the stream into the trial record, which becomes the
queryable live state; there is no separate distributed dict. Metrics time series are
WandB's job; mi-ni reads from it rather than reimplementing it (see below).

Two infrastructure constraints to keep in mind throughout, even though neither
needs solving now: Modal preempts long functions, so any trial over a few
minutes must checkpoint and resume (Phase 1); and the controlling process needs
a durable home (a dev container suspends with the laptop, a Codespace stops on
inactivity), which argues for a controller that holds no state of its own and
can resume from the registry (see "Where the controller lives").

What's missing is the connective tissue. An experiment today is a function
defined ad hoc in a notebook; its config, code version, metrics, and artifacts
are scattered and known only to the human who wrote the cell. There is no run
_registry_, no typed _experiment_ object, no way to ask "what have we tried, and
what happened?" without rereading notebooks. Closing the loop means giving
experiments an identity and a memory.

## Design principles

- Legibility. We are optimizing for the most inspectable loop, not the most
  autonomous agent. Every decision an agent makes should be reconstructable from
  durable artifacts; a human should be able to audit a week of agent activity in
  minutes.
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

[^sec]: Environment capture is an allowlist, not a dump: record package
    versions, the git SHA, and relevant config; never raw environment variables,
    so API keys and tokens can't leak into a record that gets checked in.

This is similar to MLflow's experiment/run split, but with provenance treated as
mandatory. A `Registry` provides the query surface (`list`, `filter`, `compare`,
`best`) that search strategies and analysis both depend on.

The store is scoped to a single experiment: it holds that experiment's trials
(including prior ones), not the whole portfolio. Cross-experiment comparison is
an outer-loop, human concern; keeping each registry experiment-scoped keeps it
bounded and lets it live in the experiment's own directory.

The simplest backing store that fits is _one JSON file per trial_ on a volume.
Because each worker owns its own file, there's no shared writer and no
contention; workers can write straight to the volume, so the progress queue is
needed only for _live_ streaming, not for the durable record. It's git-friendly
and needs no database. The weaknesses are cross-trial queries and atomic
multi-file updates, but at small-team scale (tens to hundreds of trials) globbing
and parsing is fine. Two alternatives for reference: sqlite gives indexed
queries and transactions but isn't git-friendly and serialises writes, so it's
better as a _derived index_ than a source of truth; and Modal's Dict/Queue
(Redis-backed, so writes are cheap) make a good _ephemeral live-state cache_
during a run, but not a durable record. Recommendation: JSON files as the truth,
optionally a Dict for live state, sqlite only if query volume ever demands it.

Whatever the backend, the `Registry` follows the `Apparatus`/`Volume` pattern
(`Local` and `Modal` implementations) and is deliberately _not_ the same volume
that holds experiment data, which is too easy to misconfigure into corruption.
Remote state syncs back to local so the checked-in record stays authoritative.

This is complementary to WandB, not a replacement. WandB is the metrics and
observability plane: live curves, the dashboard a human watches. The registry
is the control plane: what was run, under which SHA, with what budget, where the
artifacts are, and the WandB/Modal IDs so an agent can follow them. Reading
metrics back is a thin skill over those APIs (Phase 0).

### 2. Supervised execution (babysitting)

Wrap `Apparatus.amap` in a supervisor that turns the progress stream into
control. Because progress already carries `run_id`/`job_id` and step counts, the
supervisor can watch for the failure modes agents actually hit: a crash (retry,
or hand the traceback to the agent for a debugging patch, the AIDE/Sakana
auto-debug move), a stall (no progress for N seconds), or divergence (a metric
heading the wrong way → early-stop and reclaim the budget). It also handles
Modal preemption: because trials checkpoint to the `Volume`, a preempted or
restarted trial resumes from its last checkpoint rather than from zero. The
supervisor writes everything back to the trial record.

Budget enforcement is layered. The `Experiment` budget (max trials, spend) is
the coarse ceiling the supervisor watches; per-trial wall-clock belongs lower
down, as an `Apparatus` attribute (e.g. `app.w(timeout=...)`) that each backend
applies as it can (Modal's `timeout`, a signal-based kill locally), so a runaway
trial dies at the source rather than waiting for the supervisor.

This is asynchronous and event-driven, mirroring the harness's existing "babysit
a PR" pattern: the agent launches trials and yields, and is woken by progress
events rather than blocking or polling. Long runs don't monopolize the agent.
Fan-in is the thing to watch: a hundred workers writing to one queue will flood
it. The progress stream already rate-limits to one message per worker per
interval, and durable trial records go to per-trial files (above), so no single
mechanism sees a hundred concurrent writers.

Concretely, the supervisor wraps `amap` and reads two channels. `amap` yields
_final_ results; the progress stream carries _liveness and metrics_. Both are
needed: `amap` yields in input order, so one hung trial would block later
results; the supervisor relies on the progress stream, not `amap`, to notice
trouble. Early-stop is cooperative, because neither a thread-pool thread nor a
Modal call is portably killable mid-run: the trial polls a stop flag and exits
at the next checkpoint. The one new capability the apparatus must expose is a tap
on the progress stream (today it's consumed internally by the Rich display) plus
that per-job stop signal, both small, additive changes.

```python
async def supervise(app, experiment, strategy):
    history = experiment.registry                    # experiment-scoped store
    while not strategy.should_stop(experiment, history):
        configs = experiment.budget.admit(           # trim to what's affordable
            strategy.propose(experiment, history))
        if not configs:
            break
        async for trial in run_batch(app, experiment, configs):
            history.put(trial)                       # (or the worker wrote its own file)

async def run_batch(app, experiment, configs):
    live = {c.id: TrialState(c) for c in configs}    # in-memory control-plane

    async def monitor():
        async for msg in app.progress_stream():      # ProgressMessage(run_id, job_id, step, …)
            s = live[msg.job_id]
            s.observe(msg)                           # heartbeat + metric point
            if s.stalled() or s.diverging():
                s.stop.set()                         # cooperative: the trial polls this

    mon = asyncio.create_task(monitor())
    try:
        async for result in app.amap(run_trial, configs, kwargs={'exp': experiment}):
            yield Trial.finalize(live[result.id], result)
    finally:
        mon.cancel()

def run_trial(config, *, exp):
    state = load_checkpoint(exp, config)             # resume after preemption
    for step in range(state.step, config.steps):
        ...                                          # train one step
        emit_progress(step, config.steps, metrics)   # -> progress stream
        if step % CKPT == 0:
            save_checkpoint(exp, config, step)       # survive preemption
        if should_stop():                            # cooperative early-stop
            break
    return Result(id=config.id, ...)
```

### 3. The iteration loop (design and decide)

There are two nested loops here, and keeping them distinct is the key design
decision. The **inner loop** searches a _fixed_ experiment: same code, varying
configuration. The **outer loop** is the researcher's: changing the code,
defining new experiments, interpreting results, deciding what to ask next. An
LLM agent operates the outer loop; the inner loop is cheap, deterministic, and
needs no LLM at all.

The inner loop is a `Strategy`, a config proposer over a fixed experiment:

```
Strategy.propose(experiment, history) -> list[Trial configs]
Strategy.should_stop(experiment, history) -> bool
```

Ship a few concrete strategies: grid/random over the space, an Optuna-backed
Bayesian optimizer, and a config-space tree search (expand the most promising
trial, mutate its config). The `temporal` dopesheet is the natural
representation for a proposed schedule. None of these touch code; they only move
within the parameter space the experiment declares.

The outer loop is the agent harness itself (Claude Code, or a subagent it
spawns), not another `Strategy`. It does what a researcher does: reading the
registry, editing code, defining an experiment, launching a sweep (which may
delegate to a `Strategy`), reading the report, and iterating. This is why
agent-written _code_ variation lives here and not in the inner loop, and it's
the right altitude for the human checkpoint: a person reviews an experiment's
design and its report, not every config the inner loop proposes. The point of
automating this loop is to let the researcher delegate and multitask, not to
replace their judgement about what is worth trying.

### Config-space vs code-space variation

The two loops also resolve how to vary code without contention: the thing the
tree-search systems handle by mutating a shared working tree. We don't. Two
kinds of variation, kept separate:

- **Config-space** variation is the inner loop: one immutable code version (a
  git SHA), many configs, run in parallel via `amap`. No code changes, so no
  conflict; this is the common case and the default.
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
report (`mini.vis` comparison plots across trials plus the agent's written
rationale) that drops into the existing Pages pipeline. The narrative
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

Both loops need a process to drive them, but it needn't be a bespoke always-on
service; Modal is the wrong place for it, since we want compute to stay ephemeral
(heavy work runs remotely on demand, everything else on a laptop).
Instead, make the controller cheap to lose. Because the registry is the source
of truth and trials checkpoint, the controller holds no state of its own: it
proposes the next batch, launches it, and any later invocation resumes from the
registry. The loop is poll-and-resume, not blocking.

That relaxes the requirement to "the launching environment is _somewhat_
reliably available," which several setups satisfy: a workstation kept awake, a
rented cloud VM, or a Claude Code web session that wakes periodically; its
scheduled check-in or event-wake mechanism is exactly this shape, the same one
that babysits PRs. The convenient options have limits worth naming: a web or
desktop session idles out in minutes and can be killed mid-wait, and a
Codespace's idle timeout is configurable only up to four hours and can be capped
by org policy. So neither is a dependable always-on host, but both are fine as a
periodic driver of a resumable loop. The one case that genuinely wants durable
compute (trials that each run for days) is served by a _temporary_ deployment for
that experiment, not a standing one.

This only works if remote trials survive the controller sleeping, and Modal's
defaults don't give that for free. mi-ni today runs trials through an _ephemeral_
app (`app.run()`) and a client-driven `.map()`: when the laptop sleeps, Modal
detects the dropped client, stops the ephemeral app, and cancels its functions;
the work is lost and there's nothing to reconnect to. The resumable path is
`spawn()` on a deployed (or `detach`ed) function, which keeps running
independently of the client; the controller persists the returned `FunctionCall`
IDs in the registry and reconnects on wake via `FunctionCall.from_id`. The live
progress queue is sacrificial (it dies with the connection), but that's fine:
the durable result is the trial's own file and checkpoint on the volume, not the
streamed return value (which Modal retains only for about a week before
`OutputExpiredError`). So a spawn/detached execution mode is a concrete new
capability the `ModalApparatus` needs for unattended runs; the attached `.map()`
path stays the right choice while the controller is awake.

Prototype this as a separate Modal apparatus so iterating on it doesn't
destabilize the working `.map()` path; the long-term goal, though, is a single
`ModalApparatus` with detach-and-resume as a core capability, not a second class
to maintain. The convergence is clean if we get the interface right: resume
doesn't fit `amap`'s shape (a single generator that runs to completion), because
it's spawn → persist handles → poll across controller wakes. So make
spawn-and-poll the _base_ capability and let `amap` become a thin
run-to-completion wrapper over it. Folding the prototype back in is then additive
(new methods on the one apparatus) rather than a fork to reconcile, and
`LocalApparatus` stays uniform (local "spawn" is just a pollable future). With
the progress-stream tap and per-job stop signal, this is where the design pushes
hardest on the `Apparatus` abstraction, so the spawn/poll interface is worth
designing deliberately.

### Teleporting an experiment

Claude Code can _teleport_ a session between a local IDE and the web. This is the
same "the driver disappears, a fresh one picks up" problem as a sleeping
controller, with a surface change instead of a sleep; the same design carries it,
and teleport is a good test of whether we got that design right. Teleport
moves two things: the conversation and the git branch. It requires a clean
working tree (uncommitted changes are stashed), and the web side runs in a fresh
container that clones the repo. So the rule is simple: anything not committed to
the branch, or not in a remote store reconnectable by ID, does not survive the
hop. That excludes the `.mini/` local volume (gitignored scratch), the venv, the
running controller process and its in-memory state, and secrets.

Most of what's needed is already here: the registry as git-friendly JSON on the
branch travels automatically, spawned trials reconnect by `FunctionCall` ID,
checkpoints live on the Modal volume, and a stateless controller resumes from the
registry. Four teleport-specific requirements remain:

- **Commit records to the branch as the run progresses**, not just at the end,
  so live state crosses with the session rather than being stranded in `.mini/`.
- **Teleportable means Modal, not Local.** A `LocalApparatus` run keeps its data
  in gitignored `.mini/`, so it won't survive the hop; teleportable runs use the
  Modal apparatus (named/deployed app, spawn, remote volume). This is a stated
  limitation, not a bug to fix.
- **A SessionStart hook prepares the fresh web container**: run `./go install`,
  authenticate Modal and WandB from the environment's secrets, rehydrate the
  registry from JSON, and reconnect to the running app and `FunctionCall`s.
- **Configure the web environment**: Modal/WandB credentials as secrets, and a
  network policy that permits those endpoints.

One limitation to design around: because teleport needs a clean tree, only
committed state resumes cleanly; in-flight _code_ edits get stashed at the
boundary. That's fine because trials run against a recorded SHA anyway, but it
means teleporting mid-experiment wants the experiment's code already committed.

## Roadmap

Each phase is independently useful and leaves the repo in a shippable state.

**Phase 0: make a run legible.** Add `Experiment`, `Trial`, and a
`Volume`-backed `Registry`. Capture provenance (git SHA, config, environment) at
launch. Retrofit one existing notebook (`gpt_sweep.py`) to record its sweep.
_Outcome: you can ask "what did we run and what happened" after the fact._

**Phase 1: make a run observable.** Add the supervisor over `amap`: crash
retry, stall detection, metric-based early stopping, budget enforcement, and
checkpoint/resume so trials survive Modal preemption, all flowing into trial
records via the existing progress stream.
_Outcome: long sweeps babysit themselves and stop wasting compute._

**Phase 2: close the loop without an LLM.** Add the `Strategy` interface with
grid/random and Optuna backends, plus an auto-generated comparison report.
_Outcome: define an experiment, get a tuned result and a written summary: the
classic HPO loop, but on mi-ni's substrate and fully recorded._

**Phase 3: put the agent in the outer loop.** Give the agent harness the
tools and skills to do what a researcher does: read the registry, edit code,
define an experiment, launch a sweep (delegating the inner search to a Phase 2
`Strategy`), respond to supervisor events, and write the report's rationale.
The agent doesn't replace the strategies; it _uses_ them, and authors the code
they search over.
_Outcome: an agent runs an iterative experiment end to end, with a human
approving the design and reviewing the report._

**Phase 4: harden for unattended operation.** Cost and rate guardrails,
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
  fine; they're the inner loop. The agent earns its keep at the outer loop, by
  running experiments end to end (including writing the code) so the researcher
  can delegate and multitask. Phases 0–2 are tractable engineering; the open
  question is whether the agent's _end-to-end_ throughput is good enough to
  trust, which is what the Phase 4 eval harness measures.
- **Cost of autonomy.** An agent that launches GPU jobs can burn money fast.
  Budgets are in the `Experiment` object from Phase 0 for exactly this reason;
  hard guardrails land in Phase 4 before anything runs unattended.
- **The controller still needs a driver.** Holding all state in the registry
  makes the loop resumable, so it tolerates a flaky controller, but something
  must still wake periodically to advance it. The poll-and-resume design assumes
  a "somewhat reliably available" launcher; making that robust, with a sensible
  wake cadence, is design work that isn't done.

[sakana]: https://github.com/SakanaAI/AI-Scientist-v2
[agentlab]: https://agentlaboratory.github.io/
[aide]: https://github.com/WecoAI/aideml
[mlebench]: https://github.com/openai/mle-bench

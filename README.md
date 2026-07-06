> **<ruby>見<rt>み</rt>に</ruby> /mi·ni/** — _with intent to see_ [^etymology]

[^etymology]: From 見に行く (mi-ni iku), meaning "to go for the purpose of seeing something."

mi-ni is a template repository and library for doing AI research. Features:

- **Local Python notebooks** with Marimo, published to GitHub Pages
- **Remote GPU compute** at the level of functions with [Modal](https://modal.com)
- **Detached, memoized experiments** driven from a stateless CLI, so you (or an agent) can launch a run, close the laptop, and pick it up later
- **Agentic coding config** for Claude Code

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/z0u/mi-ni)

There are two ways to compute: interactive, and detached.

**Interactive.** Map a function over a sweep, right in a notebook. Swap the apparatus to change where it runs; the code stays the same:

```py
# app = LocalApparatus("my-experiment", max_workers=4)
app = ModalApparatus("my-experiment").w(gpu="L4")
metrics = list(app.map(train, sweep_configs))
app.volume.download("outputs", "local/outputs")
```

```bash
./go open ./docs/getting_started.py  # Edit in Marimo
```

[See: getting started notebook](./docs/getting_started.py).

**Detached & memoized.** For sweeps, multi-step pipelines, and long runs. Define the experiment as an importable `main(ctx)` DAG; drive and monitor it from the CLI across separate processes. Work is launched detached, and its results, progress, and errors are written to durable storage — so you can close your laptop and check back later, and so can an agent:

```py
# docs/pipeline/experiment.py
def main(ctx):
    meta = ctx.run(prepare_data)                  # one step
    return ctx.map(train, derive_configs(meta))   # a sweep that depends on it

experiment = Experiment(name="pipeline", main=main)
```

```bash
mini run docs/pipeline/experiment.py --watch   # drive to completion, live bar
mini status pipeline                            # poll later, from anywhere
```

[See: pipeline experiment module](./docs/pipeline/experiment.py).

**Report, then publish.** `report.py` is a Marimo notebook that reads the durable results from the experiment and renders them. Figures are externalized and bundled, allowing agents to view them and keeping the report light:

```python
from mini.reports import report_bundle, use_publisher
from mini.vis import themed

use_publisher(report_bundle(__file__))   # themed figures → _assets/, by name

@themed(alt_text="Final validation loss...")
def _loss_chart() -> plt.Figure: ...
```

```bash
./go export  docs/pipeline/report.py   # export the bundle locally (offline preview)
./go publish docs/pipeline/report.py   # export + mirror to the bucket (needs ./go auth)
./go serve                             # build the static site and serve it
```

At export the HTML is cleaned: progress-bar terminal sequences are collapsed, and Modal app URLs (which would leak your username) are redacted.

[See: pipeline report notebook](./docs/pipeline/report.py).

&nbsp;

<details><summary>More cool features</summary>

- [Dev container][dc] for a consistent environment, both locally and in [Codespaces][codespaces]
- ML stack ([JAX, Equinox, Pandas, etc.](pyproject.toml))
- Modern package management with [uv]
- Pre-configured for good engineering practices: tests, linting, type-checking (optional!)
</details>

&nbsp;

## Getting started

```bash
./go install  # CPU deps for local venv
./go auth     # Authenticate with Modal for remote compute
./go open docs/getting_started.py  # Open the notebook in your browser
```

For a more complete example, have a look at the [nanoGPT notebook](./docs/gpt.py).

&nbsp;

## Running experiments with an assistant

This template is set up for agentic coding (Claude Code and friends). The detached, memoized flow externalizes a run's state, results, and errors to durable storage and is driven by a stateless CLI — so an assistant can run a whole experiment for you, even across the runtime limits of a web session, by working in _wakes_: launch, stop; later check, fix, repeat.

Ask for something like:

> Write an experiment that compares X and Y, run it on Modal, watch for failures, and summarise the results in a report notebook.

The `mi-ni` skill teaches the assistant the conventions: define `main(ctx)`, drive with `mini run`, poll with `mini status`, read tracebacks with `mini logs`, and recover with `mini retry`. For a long run, it delegates launching and babysitting to a cheap monitor agent and can schedule periodic check-ins.

[codespaces]: https://github.com/features/codespaces

<details><summary>Virtual environment</summary>

The Python environment is configured when the dev container is created.

Use [uv] to add and remove packages, and to run scripts:

```bash
uv add plotly --group local
uv run python example.py
```

</details>

[dc]: https://containers.dev
[Modal]: https://modal.com
[uv]: https://astral.sh/uv

<!-- template-only -->

&nbsp;

## Contributing & licence

This project is dedicated to the public domain [^unlicense]. In your own experiments, there's no need to contribute back! The code is yours to modify as you please.

If you do want to contribute to _this template_, then fork it as usual. Before making a pull request, run:

```bash
./go check
```

[^unlicense]: Technically, the licence is the [Unlicense](https://unlicense.org), which is about as close as you can get to "do whatever you want".

<!-- /template-only -->

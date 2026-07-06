# mi-ni

> **<ruby>見<rt>み</rt>に</ruby> /mi·ni/** — _with intent to see_ [^etymology]

[^etymology]: From 見に行く (mi-ni iku), meaning "to go for the purpose of seeing something."

mi-ni is a template repository and library for doing AI research. Features:

- **Local Python notebooks** with Marimo, published to GitHub Pages with their figures served from a Hugging Face bucket
- **Remote GPU compute** at the level of functions with [Modal](https://modal.com)
- **Detached, memoized experiments** driven from a stateless CLI, so you (or an agent) can launch a run, close the laptop, and pick it up later
- **Agentic coding config** for Claude Code

Compute abstraction pattern:

```py
# app = LocalApparatus("my-experiment")
app = ModalApparatus("my-experiment").w(gpu="L4")
metrics = app.map(train, sweep_configs)
app.volume.download(...)
```

[See z0u/mi-ni](https://github.com/z0u/mi-ni).

&nbsp;

## Published notebooks

Notebooks are automatically published to GitHub Pages; their figures and other heavy assets are served from a Hugging Face bucket (repointed at build time with one `<base>` tag).

The notebooks build on each other, so they read well in order.

<!-- These URLs are rewritten to point to the published notebooks -->

### Start here

- [Getting started](./getting_started.py): map a function over a sweep from a notebook, and swap local ↔ Modal compute without changing the code.

### The detached, memoized flow

- [Pipeline](./pipeline/report.py): a multi-step experiment driven by the CLI — a prep step feeds a training sweep, and the report reads the durable results back.
- [Probe](./probe/report.py): reuses activations that a separate experiment ([acts](./acts/experiment.py)) shared through the content-addressed artifact store, by name.

### A case study at scale

- [Sweep over GPT architectures](./gpt-sweep/report.py): LayerNorm vs. hypersphere (nGPT) across learning rates. Adds hyperparameter schedules, role-based hardware routing, and artifacts published by name.
- [nanoGPT and nGPT, interactively](./gpt.py): the same models trained inline in one notebook (source only; it re-trains on every run).

### Visualization utilities

- [Themed (light/dark) plots](./themed.py): the `@themed` decorator that every report above uses for dual-mode figures.
- [Sparkline text annotations](./subline_demo.py): per-token annotations with the sibling `subline` library.

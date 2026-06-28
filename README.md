> **<ruby>見<rt>み</rt>に</ruby> /mi·ni/** — _with intent to see_ [^etymology]

[^etymology]: From 見に行く (mi-ni iku), meaning "to go for the purpose of seeing something."

mi-ni is a template repository and library for doing AI research. Features:

- **Local Python notebooks** with Marimo, published to GitHub Pages
- **Remote GPU compute** at the level of functions with [Modal](https://modal.com)
- **Agentic coding config** for Claude Code

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/z0u/mi-ni)

There are two ways to compute: interactive, and detached.

**Interactive.** Map a function over a sweep, right in a notebook. Swap the apparatus to change where it runs; the code stays the same:

```py
# app = LocalApparatus('my-experiment', max_workers=4)
app = ModalApparatus('my-experiment').w(gpu='L4')
metrics = list(app.map(train, sweep_configs))
app.volume.download('outputs', 'local/outputs')
```

[Getting started notebook →](./docs/getting_started.py)

**Detached & memoized.** For sweeps, multi-step pipelines, and long runs. Define the experiment as an importable `main(ctx)` DAG; drive and monitor it from the CLI across separate processes. Work is launched detached, and its results, progress, and errors are written to durable storage — so you can close your laptop and check back later, and so can an agent:

```py
# docs/pipeline/experiment.py
def main(ctx):
    meta = ctx.run(prepare_data)                  # one step
    return ctx.map(train, derive_configs(meta))   # a sweep that depends on it

experiment = Experiment(name='pipeline', main=main)
```

```bash
bin/mini run docs/pipeline/experiment.py --watch   # drive to completion, live bar
bin/mini status pipeline                            # poll later, from any process
bin/mini watch  pipeline                            # ...or follow it live (read-only)
```

[Pipeline example →](./docs/pipeline/report.py)

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

<details>
<summary>Notebook output cleaning</summary>

[`scripts/clean_docs.py`](scripts/clean_docs.py) runs automatically at export time and can also be triggered manually with `./go clean`. It does two things:

- **Terminal sequences** — collapses `\r`/cursor-up/erase sequences that progress bars leave behind, keeping colour codes intact.
- **Redaction** — replaces patterns that shouldn't appear in published notebooks. By default, Modal app URLs are redacted (they expose your username and app IDs). Add your own patterns to the `REDACT` list at the top of `clean_docs.py`:

  ```python
  REDACT: list[tuple[re.Pattern, str]] = [
      (re.compile(r'https://modal\.com/apps/\S+'), '[modal.com/apps/…]'),
  ]
  ```

</details>

<details>
<summary>Working with large files (the artifact store)</summary>

Large bytes don't go in Git. Instead they live in a content-addressed
**artifact store** (`mini.store`) — local by default, or a shared [Hugging Face
bucket](https://huggingface.co/docs/hub/storage-backends) when `[tool.mini] store-bucket`
is set. A step `put`s its bytes and returns a small `Artifact` handle; another
experiment or a report resolves it by content hash or by a named ref. Typically:

- training data, tokenized corpora, activation caches
- model weights and checkpoints
- report figures and data blobs (exported per report, synced to the bucket by `./go publish`)

See the `mi-ni` skill's storage reference for `put`/`get`/`publish` and report bundles.

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

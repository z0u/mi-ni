> **<ruby>見<rt>み</rt>に</ruby> /mi·ni/** — _with intent to see_ [^etymology]

[^etymology]: From 見に行く (mi-ni iku), meaning "to go for the purpose of seeing something."

mi-ni is a template repository and library for doing AI research. Features:

- **Local Python notebooks** with Marimo, with outputs stored in Git LFS and published to GitHub Pages
- **Remote GPU compute** at the level of functions with [Modal](https://modal.com)
- **Agentic coding config** for Claude Code

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/z0u/mi-ni)

There are two ways to compute, sharing one storage abstraction.

**Interactive** — map a function over a sweep, right in a notebook. Swap the apparatus to change _where_ it runs; the code stays the same:

```py
# app = LocalApparatus('my-experiment', max_workers=4)
app = ModalApparatus('my-experiment').w(gpu='L4')
metrics = list(app.map(train, sweep_configs))
app.volume.download('outputs', 'local/outputs')
```

[Getting started notebook →](./docs/getting_started.py)

**Detached & memoized** — for sweeps, multi-step pipelines, and long runs. Define the experiment as an importable `main(ctx)` DAG; drive and monitor it from the CLI across separate processes. Work is launched detached, and its results, progress, and errors are written to durable storage — so you can close your laptop and check back later, and so can an agent:

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

> Write an experiment under `docs/<name>/` that compares X and Y, run it on Modal, watch for failures, and summarise the results in the report notebook.

The assistant has skills that teach it the conventions — `mi-ni` for the library and `experiments` for this flow: define `main(ctx)`, drive with `mini run`, poll with `mini status` (never by re-running), read tracebacks with `mini logs`, and recover with `mini retry`.

If you encounter network issues, ensure the following domains are accessible from your environment (e.g. [in Claude Code](https://code.claude.com/docs/en/claude-code-on-the-web#network-access)):

```
storage.googleapis.com
modal.com
*.modal.com
*.modal-storage.com
*.modal.run
```

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

A pre-commit hook runs [`scripts/clean_docs.py`](scripts/clean_docs.py) on staged Marimo outputs. It does two things:

- **Terminal sequences** — collapses `\r`/cursor-up/erase sequences that progress bars leave behind, keeping colour codes intact.
- **Redaction** — replaces patterns that shouldn't appear in published notebooks. By default, Modal app URLs are redacted (they expose your username and app IDs). Add your own patterns to the `REDACT` list at the top of `clean_docs.py`:

  ```python
  REDACT: list[tuple[re.Pattern, str]] = [
      (re.compile(r'https://modal\.com/apps/\S+'), '[modal.com/apps/…]'),
      (re.compile(r'your-pattern'), '[replacement]'),
  ]
  ```

</details>

<details>
<summary>Working with large files (Git LFS)</summary>

This project is preconfigured to use [Git LFS](https://git-lfs.com). If you commit a matching file, it won't clog up your main Git history. By default, files in `docs/**/__marimo__/` are stored in LFS; see [`.gitattributes`](.gitattributes).

Typically, you would store _data_ rather than code in LFS:

- training data
- model weights
- visualizations (images and video)

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

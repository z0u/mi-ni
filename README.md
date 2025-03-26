# mi-ni — AI Research Template

> **<ruby>見<rt>み</rt>に</ruby> /mi·ni/** — _with intent to see_ [^etymology]

[^etymology]: From 見に行く (mi-ni iku), meaning "to go for the purpose of seeing something." This library is about small AI experiments—quick, lightweight explorations to try and see what happens.

&nbsp;

This is a template repository for doing AI research. Features:

- **Local Python notebooks**
- **Remote per-function GPU compute** [^modal]
- **Inline visualization** with remote-to-local callbacks
- **AI-assisted coding** with Copilot/VS Code


<details><summary>More cool features!</summary>

- [Dev container][dc] for a consistent environment, both locally and in Codespaces
- ML stack ([PyTorch, Polars, etc.](pyproject.toml))
- Modern package management with [uv]
- Pre-configured for good engineering practices: tests, linting, type-checking (optional!)
</details>

[^modal]: [Modal] is used for remote compute. They charge per-second, billed for the duration of your function.


## Getting started

If you want to run an experiment, make a copy of this repository. Since your project isn't a fork, you don't need to worry about keeping the code in sync, and you can add and remove Python packages as you wish.

```bash
./go install cpu  # CPU deps for local venv
./go auth         # Authenticate with Modal for remote compute
```

Then open the [Getting Started notebook](getting-started.ipynb) and try it out. Choose `.venv/bin/python3` as the kernel. For a more complete example, have a look at the [nanoGPT notebook](nanogpt.ipynb).


<details><summary>Virtual environment</summary>

The Python environment is configured when the dev container is created.

Use [uv] to add and remove packages, and to run scripts:

```bash
uv add plotly --group local
uv run python example.py
```
</details>

<details>
<summary>Restarting the language server (VS Code)</summary>

If you open a Python file before the setup is complete, you may need to restart the Python language server.

- Open a `.py` or `.ipynb` file
- Open the command pallette with <kbd>⇧</kbd><kbd>⌘</kbd><kbd>P</kbd> or <kbd>Ctrl</kbd><kbd>Shift</kbd><kbd>P</kbd>
- Run _Python: Restart Language Server_.
</details>

[dc]: https://containers.dev
[Modal]: https://modal.com
[uv]: https://astral.sh/uv


## Contributing & licence

This project is primarily released under the [Unlicense](https://unlicense.org/) (public domain). In your own experiments, there's no need to contribute back! The code is yours to modify as you please[^attrib].

If you do want to contribute to _this template_, then fork it as usual. Before making a pull request, run:

```bash
./go check
```

[^attrib]: Exception: Code in `src/experiment` is derived from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy and is subject to MIT license terms. See the [LICENSE](LICENSE) file for details.

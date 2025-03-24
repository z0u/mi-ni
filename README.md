# Mi-ni — AI Research Template

> **<ruby>見<rt>み</rt>に</ruby> /mi·ni/** — _to go and see_ • _to observe_
>
> Derived from **見に行く** (mi-ni iku), meaning "to go for the purpose of seeing something." This library is about small AI experiments—quick, lightweight explorations to try and see what happens.

This is a template repository for doing AI research. Features:

- [Dev container][dc] for a consistent environment
- ML stack ([PyTorch, Polars, etc.](pyproject.toml))
- Jupyter notebooks
- Function-level remote compute with [Modal]
- Modern package management with [uv]


## Getting started

If you want to run an experiment, make a copy of this repository. Since your project isn't a fork, you don't need to worry about keeping the code in sync, and you can add and remove Python packages as you wish.

```bash
./go install cpu  # omit `cpu` if you want the default PyTorch
./go auth         # Authenticate with Modal for remote compute
```

Then open the [demo notebook](notebook.ipynb) and try it out. Choose `.venv/bin/python3` as the kernel.


## Virtual environment

The Python environment is configured when the dev container is created. If you open a Python file before the setup is complete, you may need to restart the Python language server.

<details>
    <summary>Restarting the language server in VS Code</summary>
    <ol>
        <li>Open a <code>.py</code> or <code>.ipynb</code> file</li>
        <li>Open the command pallette with <kbd>⇧</kbd><kbd>⌘</kbd><kbd>P</kbd> or <kbd>Ctrl</kbd><kbd>Shift</kbd><kbd>P</kbd></li>
        <li>Run <em>Python: Restart Language Server</em>.</li>
    </ol>
</details>

Use [uv] to add and remove packages, and to run scripts:

```bash
uv add pydantic
uv add plotly --group local
uv run python my-experiment.py
```

[dc]: https://containers.dev
[Modal]: https://modal.com
[uv]: https://astral.sh/uv


## Contributing

In your own experiments, there's no need to contribute back! The code is yours to modify as you please.

If you do want to contribute to _this template_, then fork it as usual. Before making a pull request, run:

```bash
./go check
```

## License

This project is primarily released under the [Unlicense](https://unlicense.org/) (public domain).

**Exception:** Code in `src/experiment` is derived from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy and is subject to MIT license terms.

See the [LICENSE](LICENSE) file for details.

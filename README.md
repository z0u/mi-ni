# AI research template

This is a template repository for doing AI research. Features:

- [Dev container][dc] for a consistent environment
- ML stack ([PyTorch, Polars, etc.](pyproject.toml))
- Jupyter notebooks
- Function-level remote compute with [Modal]
- Modern package management with [uv]


## Getting started

If you want to run an experiment, make a copy of this repository. Since your project isn't a fork, you don't need to worry about keeping the code in sync, and you can add and remove Python packages as you wish.

```bash
uv sync --all-groups
uv run modal setup  # Authenticate with Modal for remote compute
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

Use `uv` to add and remove packages, and to run scripts:

```bash
uv add pydantic
uv run python my-experiment.py
```

[dc]: https://containers.dev
[Modal]: https://modal.com
[uv]: https://astral.sh/uv


## Contributing

If you want to contribute to _this template_, then fork it as usual.

Before making a pull request, run:

```bash
./scripts/check.sh all
```

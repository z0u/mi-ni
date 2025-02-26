# AI research template

This is a template repository for doing AI research. Features:

- [Dev container][dc] for a consistent environment
- ML stack ([PyTorch, Polars, etc.](pyproject.toml))
- Jupyter notebooks
- Function-level remote compute with [Modal]
- Modern package management with [uv]


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

Use `uv` to add and remove packages:

```bash
uv add pydantic
```

Regarding notebooks: Jupyter itself is not installed, because your editor may provide its own notebook UI. However, `ipykernel` is installed into the virtual environment when the dev container is created. Therefore, you should choose `.venv/bin/python3` as the kernel when you run a cell. You might need to restart the Python language server after that.


## Model training with remote compute

This project optionally uses [Modal] for remote GPU resources. You don't need to configure it (that's in the code), but you do need an account.

```bash
uv run modal setup  # Authenticate
uv run modal run main.py
```

A demo is available in [notebook.ipynb](notebook.ipynb) and [main.py](main.py).


[dc]: https://containers.dev
[Modal]: https://modal.com
[uv]: https://astral.sh/uv

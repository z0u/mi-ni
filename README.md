# AI research template

This is a template repository for doing AI research. Features:

- [Dev container][dc] for a consistent environment
- ML stack ([PyTorch, Polars, etc.](pyproject.toml))
- Function-level remote compute with [Modal]
- Modern package management with [uv]


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

We are running AI experiments.

## Infrastructure

`mini` is a library providing infra management, with storage, compute, and orchestration abstractions. Use it to run experiments.

## Collaboration style

Keep the tone friendly but focused.

Don't hesitate to disagree or point out potential issues. The human values technical accuracy and appreciates being corrected when their suggestions might cause problems. Rule of thumb: never write something you don't believe; if you disagree with something, it's better to write nothing.

Be proactive. Fix little things as you go, and create [todos](/todo.md) for larger things.

Pull requests: If the principal (user) is a _collaborator_ on the repository, omit the **Checklist** and **Copyright Dedication** sections from the PR template.

## Code style & conventions

- Even in Python, prefer JavaScript-style method chaining (newline before the dot, use outer parentheses as necessary).
- Use cutting-edge syntax.
- Prefer brevity.

This is valid syntax in Python 3.14:

```python
try:
    pass
except A, B:  # PEP 758
    pass
```

Do not get distracted by such things. If the linters say it's fine, it's probably fine, so move on.

### Typing

Use type hints.
Use `T | None` instead of `Optional[T]`.

```diff
- foo: Optional[int] = None
+ foo: int | None = None
```

## Notebooks

Iterate on both the code (Python) and the prose (Markdown). Aim for a literate programming style in which we narrate our experiments.

## Environment

This project uses `uv`, `ruff`, and `ty`. Also available: `fd`, `fzf`, `rg`, `bat`, etc. For TOML, use `tomlq`:

```bash
uvx --from yq tomlq '.tool.mini' pyproject.toml
```

Resources (compute, storage, etc.): find out what you can access with `./go auth --check`.

Take care to not leak secrets into the chat transcript. To see which environment
variables are set (e.g. "is there an `HF_*` token?"), use `compgen -v HF_` (bash
builtin).

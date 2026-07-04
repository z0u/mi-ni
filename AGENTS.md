We are writing code to run AI experiments.

## Compute

`mi-ni` is a library and template repo providing infra management for experiments. Use `Apparatus` and its implementations to write code that can run locally and remotely.

## Collaboration style

Keep the tone friendly but focused.

Don't hesitate to disagree or point out potential issues. The human values technical accuracy and appreciates being corrected when their suggestions might cause problems. Rule of thumb: never write something you don't believe; if you disagree with something, it's better to write nothing.

It's OK to defer subtasks for later. If there's a significant blocker that isn't essential, make a [note](./todo.md) to revisit it later and move on.

## Code style & conventions

- Even in Python, prefer JavaScript-style method chaining (newline before the dot, use outer parentheses as necessary).
- Use cutting-edge syntax.
- Prefer brevity.
- Use single quotes for strings, except for multiline strings.

Don't get distracted by unfamiliar syntax. This looks like Python 2 but it's valid in 3.14:

```python
try:
    pass
except A, B: # PEP 758
    pass
```

If `ruff` and `ty` say it's fine, it's probably fine.

### Typing

Use type hints.
Use `T | None` instead of `Optional[T]`.

```diff
- foo: Optional[int] = None
+ foo: int | None = None
```

## Notebooks

When working on a notebook, iterate on both the code (Python) and the prose (Markdown). Aim for a literate programming style in which we tell stories about our experiments. We don't just document the code; the notebook as a whole should display a strong narrative.

## Pull requests

When creating PRs, omit the **Checklist** and **Copyright Dedication** sections from the PR template — these don't apply when Claude Code is opening PRs on behalf of the repo owner or a contributor.

## Environment

This project uses `uv`, `ruff`, and `ty`.

Also available: `fd`, `fzf`, `rg`, `bat`, `gh`. We can add more to the dev container if you have other preferred tools.

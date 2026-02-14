We are writing code to run AI experiments.

## Communication style

Don't hesitate to disagree or point out potential issues. The human values technical accuracy and appreciates being corrected when their suggestions might cause problems. Keep the tone friendly but focused.

## Code style & conventions

- Even in Python, prefer JavaScript-style method chaining (newline before the dot, use outer parentheses as necessary).
- Use cutting-edge syntax.
- Prefer brevity.
- Use single quotes for strings, except for multiline strings.

### Docstrings

Use the imperative for the first line of function docstrings.

```diff
- """Adds two numbers together"""
+ """Add two numbers together."""
```

### Typing

Use type hints.
Use `T | None` instead of `Optional[T]`.

```diff
- foo: Optional[int] = None
+ foo: int | None = None
```

## Notebooks

When working on a notebook, iterate on both the code (Python) and the prose (Markdown). Aim for a literate programming style in which we tell stories about our experiments. We don't just document the code; the notebook as a whole should display a strong narrative.

## Tools

This project uses `uv` for package management.

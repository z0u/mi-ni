---
name: style-py
description: Python style and typing conventions for this repo — method chaining, modern syntax, where to put type hints, and the literate programming standard for reports. Use when writing or reviewing any Python.
---

House style, in three lines: chain your calls, use the newest syntax the toolchain accepts, and keep it short.

## Shape of the code

Prefer JavaScript-style method chaining, even in Python. Put the newline _before_ the dot, and wrap the whole expression in parentheses when you need to:

```python
result = (
    frame
    .filter(pl.col("lr") > 0)
    .group_by("arch")
    .agg(pl.col("val_loss").mean())
)
```

Prefer brevity. A shorter version that reads the same is the better version.

## Cutting-edge syntax is encouraged

We track new Python releases and use what they give us. For example, PEP 758 multi-exception `except` without parentheses is valid in 3.14:

```python
try:
    pass
except A, B:  # PEP 758
    pass
```

If something looks unfamiliar, check the linters rather than rewriting it. `ruff` and `ty` are the arbiters: if they're happy, the code is fine.

## Typing

Annotate, to give the type-checker something to catch and the IDE something to complete.

Use `T | None`, never `Optional[T]`:

```diff
- foo: Optional[int] = None
+ foo: int | None = None
```

You don't need annotations everywhere. Put one wherever inference would otherwise stall — usually the point where a value first enters the code. That single annotation then carries through everything downstream.

A literate script's cells are ordinary top-level code in one shared namespace — there are no generated function signatures to keep in sync, so annotate the same way you would in any module: wherever inference would otherwise stall, usually where a value first enters the code. A name reused across cells is typed by flow, the same as any reassigned local.

### Matplotlib axes

`plt.subplots()` returns `tuple[Figure, Any]`, because the second element's shape depends on the arguments. `cast()` it to the alias that matches what you asked for:

```python
from mini.vis import AxesRow
fig, axes = plt.subplots(1, 3, ...)  # 1D
axes = cast(AxesRow, axes)

from mini.vis import AxesGrid
fig, axes = plt.subplots(2, 3, ...)  # 2D
axes = cast(AxesGrid, axes)
```

## Literate scripts

Our experiments and reports ship code and prose together, as literate scripts (`mini.lit`): the only report form. Aim for literate programming: the Markdown should explain what the next cell does and why, so the report reads as an argument rather than a script with captions.

A literate script is a plain module with a `# title:` header: a top-level string is prose (an f-string where it quotes a value), and the code between two prose strings is a cell. Everything is ordinary Python, so annotate as you would in a module; there are no generated signatures. The conventions that come from the form:

- Prose is the cell boundary. A `# %%` line splits a cell where prose would not fit: two values shown back to back, or a slow loader kept apart from the fast plot beneath it so an edit to the plot re-runs the plot alone.
- An f-string doubles every literal brace, so `\frac{a}{b}` becomes `\frac{{a}}{{b}}` once the paragraph quotes a value. Keep equations in plain (`r"""…"""`) paragraphs and quote values in the paragraph beside them.
- A results table is a helper returning `<table class="report-table">` as the cell's last expression; no scroll wrapper, the table is its own scroll box on a narrow screen.
- Only a cell's last expression is displayed. A loop that builds figures ends the cell with the joined string (`"\n\n".join(...)`); a table is a helper returning HTML as the last expression. The runner refuses a displayable value anywhere else in a cell, so a stray one is an error rather than a silent gap.
- Every top-level string is prose, so a variable docstring under a constant would weave as a paragraph. Write it as a comment (`./go lint` flags the slip).
- One namespace, top to bottom: no `_private` cell names, and a name may be reused across cells (ty types by flow). Precompute joined lists as strings in the cell rather than in the prose field.
- `if cond: stop("…")` ends a preregistration early; the prose below still renders with pending marks for what it cannot evaluate.
- Slow work goes under `@memo` (outside `@themed` for a figure), with anything that should invalidate the cache, the alt text included, passed as an argument.


See the `style-fig` skill for figure and results-table conventions, and `docs/README.md` for file-type and publishing rules.

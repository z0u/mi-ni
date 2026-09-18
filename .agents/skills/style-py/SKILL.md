---
name: style-py
description: Python style and typing conventions for this repo — method chaining, modern syntax, where to put type hints given that Marimo generates cell signatures, and the literate programming standard for notebooks. Use when writing or reviewing any Python.
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

### Marimo cells

Marimo generates cell function signatures, so a parameter is bare unless the cell that _defines_ that value annotated it. Where the parameter is bare, inference has nothing to work from at the top of the cell. Annotate the first local binding and the rest of the cell follows:

```python
@app.cell(hide_code=True)
def _(ARCHS, curve):
    _archs: list[str] = [a for a in ARCHS if a != "baseline"]
    _resp: dict[str, tuple[np.ndarray, float, float]] = {a: curve(a) for a in _archs}
```

Annotate a _public_ name and Marimo copies that annotation onto the parameter list of every cell downstream, the next time it saves the file:

```python
@app.cell(hide_code=True)
def _(_sv):
    sv_trials: dict[int, dict] = {t["trial"]: t for t in _sv["trials"]}
    return (sv_trials,)


@app.cell(hide_code=True)
def _(sv_trials: dict[int, dict]):  # Marimo propagated this annotation
    ...
```

Leave them alone: if you add or change them, Marimo will regenerate them and cause churn in Git. Annotate the definition instead.

`marimo check --fix <file>` applies the rewrite from the CLI, and is authoritative over signatures: it fills in missing annotations, corrects wrong ones, and removes any with no annotated definition behind them. It runs automatically on edit (a `PostToolUse` hook) and on commit (via lint-staged), so this mostly self-corrects.

`./go annotations [path...]` names the public cell variables that are still bare, so you can see a notebook's share before you start: `./go annotations docs/gpt-sweep`. It's advisory — a worklist, not a gate — and it reports names bound by unpacking (`a, b = ...`) separately, since Python has no syntax to annotate those and the fix is to split the statement instead.

Naming:

- Symbols that are cell-local must start with `_`, or Marimo will complain.
- Symbols within nested functions should usually not start with `_`.

```python
@app.cell(hide_code=True)
def _():
    def _foo(x: int) -> int:
        y = x + 1
        return y
```

Utilities and setup:

- In general, put imports and constants in a setup cell.
- Put utility functions in their own reusable cells. Don't put them in the setup cell, or editing a function would invalidate every cell in the notebook.

```python
with app.setup(hide_code=True):
    # This is the "setup" cell
    from mini.reports import report_bundle, use_publisher

    use_publisher(report_bundle(__file__))

    LAYER_NAMES = ["emb", "1", "2", "3", "4"]
    """A docstring for a constant."""

    None  # Prevent the docstring from rendering


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class Curves:
    data: tuple[dict[str, dict[str, np.ndarray]], np.ndarray]

    def stat(self, key: str, arch: str) -> np.ndarray:
        return np.array([r[key] for r in self.data[arch]], float)


@app.function(hide_code=True)
def load_curves() -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    # This is a "reusable function" cell
    ...
    return data


@app.cell(hide_code=True)
def _():
    curves: Curves = Curves(data=load_curves())  # annotate, so downstream cells see the type
    return (curves,)


@app.cell(hide_code=True)
def _(curves: Curves):
    _stat = curves.stat(...)  # now properly typed
    ...
    return
```

Method defaults and annotations may read the setup cell, and the class body and its methods are type-checked like any module-level class. Annotating the instance is what carries that to the call sites: Marimo copies `Curves` onto every downstream signature, and `ty` then catches a misspelled method or a wrong argument in the cells that use it. Leave the instance bare and those cells go unchecked.

This is for a bundle several cells share. A `_plot()` closure inside one figure cell stays where it is — it has one caller, and the state it reads is right above it.

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

## Notebooks and literate scripts

Our experiments and reports ship code and prose together: as literate scripts (`mini.lit`, the form reports are moving to) or as Marimo notebooks. Iterate on both. Aim for literate programming: the Markdown should explain what the next cell does and why, so the report reads as an argument rather than a script with captions.

### Literate scripts

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

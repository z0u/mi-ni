---
name: style-md
description: |
  Syntax conventions for Markdown, and renderer-specific pitfalls to avoid. Read
  before editing text in .md files, Marimo notebooks, and GitHub issues.
---

Wrapping: keep each paragraph on one line — very wide — and let the editor soft-wrap it; don't hard-wrap at a fixed column. Exceptions below. For paragraphs that contain landmarks, like inline lists, put the landmarks after a single newline:

> This is a paragraph with an inline list:
> (a) Foo,
> (b) Bar.

## Links

A target starting with `/` resolves from the **repository** root, not the filesystem or domain root — GitHub rewrites it to the current ref, and VS Code resolves it against the workspace root. Prefer that form whenever a link leaves its own directory: `/eng/determinism.md` says where it lands, where `../../../../eng/determinism.md` only says how far to climb. Keep `./sibling.md` for a file alongside, and `../` where the relationship is the point — a `references/` doc pointing at the `SKILL.md` that owns it, or one backlog set pointing at another.

An `#anchor` into a heading works in both places, since GitHub and the published site slug a heading the same way. Add an explicit `<span id="user-content-..."></span>` to a heading only when the link has to survive the heading being reworded — a link from a published post, say.

`./go links` reports a target whose file is missing, an `#anchor` that no longer matches a heading. It's part of `./go check` and gates CI, so run it after renaming a heading or moving a file — the moved file gives a dead link, but the renamed heading is the quiet one, since the link still opens the file at the top and the reader lands somewhere plausible.

## GitHub

In pull requests and issues, single newlines are retained — whereas in `.md` files they are collapsed. So don't hard-wrap paragraphs in issues and PRs.

Math expressions are OK; slight preference for plain Unicode because it's easier to copy.

## Marimo

Marimo has some Markdown extensions. Consider using `details` and footnotes for asides:

```py
mo.md("""
Main content with an inline footnote,[^note] and so on.

[^note]: Renders at the end of the cell. Footnote numbers are cell-local (they restart).

/// details | Title
Some backstory.
///
""")
```

Other admonition types and their icons: `details` (folds, unobtrusive), `admonition` (unadorned), `note` ℹ️, `tip` 💡, `important` 💬, `warning` ⚠️, `error` 🛑. The `| title` is optional, except for `details`.

Plain Markdown cells are visible as soon as the notebook opens, and contribute to the TOC. But Markdown cells that use string interpolation, or anything other than a plain `mo.md("literal string")`, are not rendered until it's their turn in the DAG. Therefore, headings and their following introductory paragraph should be placed in plain Markdown cells; otherwise the document will be hard to navigate. So in landmark cells: no f-strings, and no `str.replace()`.

Math expressions, for consistency with formulas. Unicode can be used where it's cumbersome to use math mode, e.g. in embedded HTML.

Text-wrapping.
- Beware of interpolated f-strings that would put special syntax at the start of a line. A line that starts with `{value:d}. Next sentence` will render as an ordered list, even if `value` is not 1.
- Don't hard-wrap a line inside an inline code span  `` ` `` or math expression `$`. A wrapped span might start the next line with block syntax, so a hex code in an expression like ` #f78` renders as a heading, and some renderers break the span entirely. Rewrap the surrounding prose so the whole span sits on one line.

But _do_ use multiline strings; these are automatically `dedent`ed:

```patch
      mo.md(
-       "Sometimes we write Markdown in Python, e.g. when working in a Marimo notebook. "
-       "In that case, prefer multiline strings rather than using one string per "
-       "hard-wrapped line. Use dedent and f-strings as needed."
+       """
+     Sometimes we write Markdown in Python, e.g. when working in a Marimo notebook. In that case, prefer a multiline string over one string literal per hard-wrapped line. Use dedent and f-strings as needed."""
      )
```

Multiline strings are also supported by the `@themed(..., alt_text=..., caption=...)` decorator (see `style-fig`).

### Interpolation and indentation

Marimo dedents a cell's source when it stores and runs it, and it never touches the inside of a string literal. So an interpolated value keeps whatever indentation was baked into it, and once the surrounding Markdown has been dedented to column 0, four leading spaces in that value make the block a code fence — the table you built renders as its own raw HTML:

```patch
- _r = f"{prose(res)}\n\n    {table(res)}"
- mo.md(f"""
-     {_r}
- """)
+ mo.md(rf"""
+ {prose(res)}
+
+ {table(res)}
+ """)
```

Two rules follow. Build the Markdown inline in the cell that displays it, rather than assembling pre-formatted fragments in one branch and dropping them into a template in another — that puts the indentation somewhere you can't see it. And let every interpolated value be flush left, with no leading whitespace of its own.

### The shape of a section

A section of a report reads best as three cells: the heading on its own with zero interpolation (so it renders as soon as the notebook opens and reaches the TOC), then the background and prediction, then the results. Guard the results cell with a raise rather than `mo.stop`:

```python
@app.cell(hide_code=True)
def _(res):
    if res is None:
        raise mo.MarimoStopError(mo.md(RESULTS_TO_COME))
    mo.md(rf"""
    **Results.** {h1_prose(res)}

    {h1_figure(res)}
    """)
```

The raise stops the cell the same way `mo.stop` does, and it narrows `res` for everything below, which `mo.stop` (a plain call returning `None`) cannot. Keep the placeholder itself in the setup cell, so every unfinished section shows the same admonition.

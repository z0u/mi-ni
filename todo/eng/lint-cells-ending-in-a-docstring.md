---
status: done
tags: [lint, reports]
opened: 2026-09-01
closed: 2026-09-01
---
# Lint for notebook cells that end in a docstring

Marimo renders a cell's last expression as that cell's output, and a variable docstring is an expression statement. So a setup cell whose last line is a docstring attached to a constant publishes that string as a stray paragraph at the top of the report. It is invisible in the source and in edit mode with `hide_code=True` — the only place it shows is the rendered page.

Two reports had it (d2.1 and ex-2.1.12); both now end their setup cell with a bare `None` and a comment, and the `style-py` example does too. That is the fix, but nothing stops the next one.

The check is a short AST pass: for each `with app.setup` block and each `@app.cell` function, flag a body whose last statement is an `ast.Expr` wrapping a string constant. `scripts/unannotated_cell_vars.py` is the model for a notebook AST script, and `scripts/lint.sh` is where it would hang. Ruff can't do it — B018 is the matching rule, but it is ignored for `docs/*.py` precisely so variable docstrings are allowed, and it wouldn't know which one is last anyway.

## Notes

**2026-09-01, tech debt** — Landed as `scripts/trailing_cell_docstrings.py`, inside `./go lint` (so it gates a merge, unlike the advisory `./go ann`) and separately runnable as `./go strays [...paths]`. Verified against a reintroduced leak in d2.1, which it reports with a file:line the editor links on.

Which cells can leak was settled by exporting a probe notebook and grepping the HTML for each shape rather than by reading Marimo's compiler, which was worth doing — one shape was a surprise. `with app.setup` leaks its last statement as written; `@app.cell` leaks the last statement *before* the generated `return`, whether that return is bare or carries values, so the check has to drop the return before looking; `@app.function` and `@app.class_definition` never leak, since their body is ordinary local scope. A trailing bare name or call is deliberately left alone — that is how a cell shows a figure.

Writing it turned up one more instance of the same shape, in the script's own module docstring: the first draft quoted a literal triple-quote to describe the pattern, which closed the docstring early.

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/lint-cells-ending-in-a-docstring.md`](https://github.com/z0u/sca2/blob/main/todo/eng/lint-cells-ending-in-a-docstring.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on; the two reports were that project's d2.1 and ex-2.1.12.

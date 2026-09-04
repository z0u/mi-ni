---
status: done
tags: [style, tooling]
opened: 2026-08-11
closed: 2026-08-11
---
# Un/rewrap all multiline prose strings

Widened code lines to 120 (ruff already used that width; only `.devcontainer/marimo.toml`'s own formatter setting was still 79) and unwrapped prose paragraphs to one line each — Markdown files, report `mo.md()` cells, and docstrings across `src/`, `scripts/`, and `tests/` — so the editor soft-wraps instead of the file carrying stale line breaks. A hard break is kept immediately before an inline landmark like `(a)`/`**(a)**`, per `style-md`. Done with a mechanical line-join script (not an LLM rewrite), so no words moved — verified by whitespace-normalized diffs, HTML-render diffs before/after, `check-templates` (report cells), and the full test suite. Two real edge cases the script had to learn to leave alone: Google-style `Args:`/`Returns:` docstring sections (a field list, not prose) and RST `::` literal code blocks (the primary code-example convention in `src/`) — the first pass flattened one such example into invalid Python before it was caught in review, which is the reminder to actually look at the diff rather than trust the checks alone.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/unwrap-multiline-prose-strings.md`](https://github.com/z0u/sca2/blob/main/todo/eng/unwrap-multiline-prose-strings.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

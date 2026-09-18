---
status: partial
tags: [tooling, reports, publishing, notebooks]
opened: 2026-09-17
---
# Literate documents without a notebook runtime (`mini.lit`)

A report needs four things: prose and Python side by side, Markdown and code cells, HTML output in our own styles, and fast re-renders (cached computation, tens of milliseconds while editing and under a second from a fresh process). It does not need cells that run out of order, a browser-side editor, or a runtime the page loads from a CDN. Our reports use a thin slice of Marimo's API (`mo.md` and `mo.stop`, over a few thousand lines of helpers), and much of the tooling around them (`report.css`, `md.css`, `clean_marimo_md.py`, `export_report_md.py`, the offline render skill) exists to bridge the rest.

`mini.lit` is the prototype: a plain Python file whose `# %%` comments start cells and whose top-level string literals are prose (or the same thing as Markdown with ```` ```{python} ```` fences), prose rendered as Jinja against the cells' namespace (interpolation, `{% for %}` tables, helper calls), the same pymdownx Markdown dialect as `mo.md`, a `@memo` cache keyed by `mini.memo`'s source and input fingerprints, `stop()` with pending marks for a preregistration, a watch-and-reload server, and Markdown, HTML and PDF outputs from one pass. Demos in `docs/lit/`; `./go lit render|serve`.

Open, in rough order:

- Settle on one spelling. The Markdown form came first and gives prose-first editing; the Python form was added because the Quarto extension gives fences highlighting but no language server, and a plain `.py` gets ruff, ty, go-to-definition, and VS Code's "Run Cell" for nothing (ty already caught a `None` subscript in the demo). The runner, cache, and page are indifferent; only `parse` differs. Dropping the `.md` parser is about 60 lines and four tests once the `.py` form has carried a real report.
- Port one real report (an sca2 experiment) and time it end to end, with figures through `@memo`. The prototype's numbers are from the two demos only.
- Publishing: the woven output is already a bundle (`index.html` + `_assets/`), so `./go publish` and `build_site` should accept a lit document beside a notebook export. The site build currently skips any `.md` under `docs/` that contains a cell.
- Markdown output: figures come through as the `<figure>` HTML `themed` emits; the `report-render` skill wanted `![alt](path)`, so `themed` should take a plain-Markdown target.
- A theme toggle on the page (the figures and swatches already honour `body[data-theme]`), self-hosted fonts, and a table-of-contents option.
- The pending mark after `stop()` shows the root name (`summary`), not the expression (`summary.m_line`); Jinja's undefined does not carry the path.
- The cache is project-wide and keyed by function name and inputs, so two documents defining a same-named cell function with the same inputs share a key and thrash the on-disk record (never serving each other's value, since the source fingerprint differs). Key by document as well if that turns up.

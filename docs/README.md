# docs/

This directory contains executable experiment notebooks and source files for the
project site. The site is built by `./go build` into `_site/`.

## File types

**Marimo notebooks** (`.py`) are the primary content — and the *only* thing in Git;
the exported HTML is never committed. `./go publish` exports each notebook to a
self-contained bundle (`index.html` + named-keyed `_assets/`) and mirrors it to the
HF bucket under `exports/<key>/`, where `<key>` is the notebook's docs-relative path
without suffix (`docs/getting_started.py` → `getting_started`, `docs/foo/bar.py` →
`foo/bar`) — except a notebook named `report.py` takes its directory as the key
(`docs/foo/report.py` → `foo`), so a one-report experiment publishes at `foo/` rather
than the redundant `foo/report/`. `./go build` then assembles `_site/` from the synced
bundles, serving each report at `_site/<key>/index.html` (URL `<key>/`). With no bucket, the build *localizes*
from `.mini/exports/` (produced by `./go export`) so it works offline.

**Markdown files** (`.md`) are converted to HTML and written to `_site/` at the
same relative path. Links to a report's `.py` are automatically rewritten to its
rendered `<key>/` page. This `README.md` is excluded from the build.

**Other assets** (images, SVGs, etc.) are copied as-is into `_site/`.

## Structure

```
docs/
├── README.md                This file (excluded from build)
├── index.md                 Built as _site/index.html
├── getting_started.py       Marimo notebook → exported bundle, served at _site/getting_started/
└── pipeline/                A heavier experiment, split into definition + report
    ├── experiment.py        Importable main(ctx) DAG — not a notebook, so the build ignores it
    └── report.py            Marimo notebook → served at _site/pipeline/
```

Exported bundles live (gitignored) under `.mini/exports/<key>/` locally; their durable
home is the bucket. Nothing under `docs/` holds generated HTML.

Heavier or multi-step experiments live in a subdirectory as an importable
`experiment.py` (the definition, driven by the `mini` CLI) plus a `report.py`
notebook (reads durable results and publishes). A plain `.py` that isn't a Marimo
notebook is ignored by the build, so the definition module never lands on the
site. See the `mi-ni` skill (authoring, running & monitoring).

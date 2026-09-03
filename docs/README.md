# docs/

This directory contains executable experiment notebooks and source files for the project site. The site is built into `_site/` — by `./go preview` locally, or `./go site` in CI.

## File types

### Marimo notebooks

Notebooks (`.py`) are the primary content, and the only thing in Git; the exported HTML is never committed.

`./go publish` exports each notebook to a self-contained bundle (`index.html` plus a name-keyed `_assets/`), and mirrors it to the HF bucket under `exports/<key>/`. The key is the notebook's docs-relative path without the suffix, so `docs/overview.py` becomes `overview` and `docs/foo/bar.py` becomes `foo/bar`. A notebook named `report.py` is the exception: it takes its directory as the key, so `docs/foo/report.py` becomes `foo`, and a one-report experiment publishes at `foo/` rather than the redundant `foo/report/`.

Publishing also records the commit sha that the bundle landed as into `publish.lock`. Commit that file. The site serves each report at its pinned revision, so a publish from a branch deploys nothing until the pin reaches main; the PR preview serves the branch's pins meanwhile.

Forgetting to publish is caught rather than done for you, in two places. The push hook ([`pre-push-check.sh`](../.claude/hooks/pre-push-check.sh)) blocks a push that changed a report without moving its pin, and CI's `Reports published` step repeats the check on the PR. Both run [`scripts/unpublished_reports.py`](../scripts/unpublished_reports.py) — a git diff against the base branch, compared with `publish.lock` — so neither needs the store, a render, or a write token. The publish itself stays with you, in the session that already has a warm store.

Three ways past it, in rising order of permanence: `git push --no-verify` gets a push out now (CI still flags it); the `skip-publish-check` label settles it for one PR; and `# mini:manual-publish` in a notebook (see `mini.reports`) opts that report out for good, for one you'd rather publish on your own schedule.

`./go site` (CI) then assembles `_site/` from the pinned bundles, serving each report at `_site/<key>/index.html`, with the URL `<key>/`. `./go preview` assembles the same site locally: it exports stale reports to `.mini/exports/` and copies their assets beside the HTML, so it works offline.

### Markdown files

Markdown (`.md`) is converted to HTML and written to `_site/` at the same relative path. Links to a report's `.py` are rewritten to its rendered `<key>/` page. This `README.md` is excluded from the build.

### Other assets

Images, SVGs, and the like are copied as-is into `_site/`.

### Shared report styles

[`report.css`](./report.css) is one stylesheet for cross-report polish: centering narrow figures, `.sw` color swatches (see the `style-fig` skill), `.report-table` headings, and `.report-subline-row`.

Each report points at it with `marimo.App(css_file="…/report.css")`, so it shows live in edit mode and bakes into the export. The build re-inlines it from source as well (`mini.reports.set_report_styles`), so editing `report.css` restyles every published report without re-exporting any notebook. Keep it small and selector-scoped, since it layers on top of Marimo's own CSS.

## Structure

```
docs/
├── README.md                This file (excluded from build)
├── report.css               Shared report stylesheet (baked via css_file + re-inlined at build)
├── publish.lock             Export key → pinned publish-tier revision (written by ./go publish)
├── index.md                 Built as _site/index.html
├── getting_started.py       Marimo notebook → exported bundle, served at _site/getting_started/
└── pipeline/                A heavier experiment, split into definition + report
    ├── experiment.py        Importable main(ctx) DAG — not a notebook, so the build ignores it
    └── report.py            Marimo notebook → served at _site/pipeline/
```

Exported bundles live (gitignored) under `.mini/exports/<key>/` locally; their durable home is the bucket. Nothing under `docs/` holds generated HTML.

`./go render <notebook>` writes a third thing, for reading rather than serving: the same document as plain Markdown at `.mini/renders/<key>.md`, figures as `![alt](…)` links to the files on disk. Same key, also gitignored, also regenerated on demand — it skips a report no older than its last render, and `--force` overrides that. See the `report-render` skill.

Heavier or multi-step experiments live in a subdirectory as an importable `experiment.py` (the definition, driven by the `mini` CLI) plus a `report.py` notebook, which reads durable results and publishes. A plain `.py` that isn't a Marimo notebook is ignored by the build, so the definition module never lands on the site. See the `mi-ni` skill for authoring, running, and monitoring.

# docs/

This directory contains executable experiment notebooks and source files for the
project site. The site is built by `./go build` into `_site/`.

## File types

**Marimo notebooks** (`.py`) are the primary content. When a notebook is
executed, Marimo captures the output and exports it as an HTML file in a
`__marimo__/` subdirectory. For example, executing `docs/getting_started.py` in
Marimo produces `docs/__marimo__/getting_started.html`.

The build script picks up all HTML files from `__marimo__/` subdirectories and
copies them into `_site/`, preserving the directory structure relative to
`docs/`. A notebook at `docs/foo/bar.py` would be exported to
`docs/foo/__marimo__/bar.html` and end up at `_site/foo/bar.html`.

**Markdown files** (`.md`) are converted to HTML and written to `_site/` at the
same relative path. Links to `.py` files are automatically rewritten to `.html`
in order to point to notebook output. This `README.md` is excluded from the
build.

**Other assets** (images, SVGs, etc.) are copied as-is into `_site/`.

## Structure

```
docs/
├── README.md                This file (excluded from build)
├── index.md                 Built as _site/index.html
├── getting_started.py       Marimo notebook (source, excluded from build)
└── __marimo__/              Marimo output
    └── getting_started.html Built as _site/getting_started.html
```

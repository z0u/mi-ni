"""
Literate documents without a notebook runtime: a plain ``.py`` whose top-level strings are prose and whose code between them is a cell (or Markdown with ``{python}`` fences), Jinja in the prose, and a disk cache for the slow parts.

    python -m mini.lit render docs/foo/report.py      # → .mini/lit/foo/index.html (+ index.md)
    python -m mini.lit serve  docs/foo/report.py      # watch, re-weave on save, live reload

See :mod:`mini.lit.document` for the format and execution model, :mod:`mini.lit.caching` for the cache, and :mod:`mini.lit.page` for the Markdown dialect.
"""

from mini.lit.document import Runner, Stop, is_literate_document, parse, stop
from mini.lit.caching import cache_dir, memo, set_cache_dir
from mini.lit.render import render, to_pdf

__all__ = [
    "Runner",
    "Stop",
    "parse",
    "is_literate_document",
    "stop",
    "memo",
    "cache_dir",
    "set_cache_dir",
    "render",
    "to_pdf",
]

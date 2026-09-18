"""
Literate documents without a notebook runtime: Markdown with ``{python}`` cells, Jinja in the prose, and a disk cache for the slow parts.

    python -m mini.lit render docs/foo/report.md      # → .mini/lit/foo/index.html (+ index.md)
    python -m mini.lit serve  docs/foo/report.md      # watch, re-weave on save, live reload

See :mod:`mini.lit.document` for the format and execution model, :mod:`mini.lit.caching` for the cache, and :mod:`mini.lit.page` for the Markdown dialect.
"""

from mini.lit.document import Runner, Stop, parse, stop
from mini.lit.caching import cache_dir, memo, set_cache_dir
from mini.lit.render import render, to_pdf

__all__ = ["Runner", "Stop", "parse", "stop", "memo", "cache_dir", "set_cache_dir", "render", "to_pdf"]

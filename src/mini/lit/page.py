"""
Markdown to HTML, and the page around it.

The Markdown dialect is python-markdown with the pymdownx extensions (tables, footnotes, ``///`` admonitions and details, arithmatex math, superfences, highlight, tilde/caret/mark, attribute lists), the dialect the reports are written in. The page is one HTML file: our stylesheet inline, the body in ``<main class="lit">``, and KaTeX pulled from a CDN only when the document has math.
"""

from __future__ import annotations

import html
import re
from functools import cache
from pathlib import Path

import markdown
from pygments.formatters.html import HtmlFormatter
from pymdownx.slugs import slugify

__all__ = ["to_html", "page", "render_fragment"]

CSS_PATH = Path(__file__).with_name("lit.css")

EXTENSIONS = [
    "tables",
    "footnotes",
    "attr_list",
    "md_in_html",
    "sane_lists",
    "toc",
    "pymdownx.arithmatex",
    "pymdownx.superfences",
    "pymdownx.highlight",
    "pymdownx.inlinehilite",
    "pymdownx.tilde",
    "pymdownx.caret",
    "pymdownx.mark",
    "pymdownx.tasklist",
    "pymdownx.blocks.admonition",
    "pymdownx.blocks.details",
    "pymdownx.blocks.caption",
    "pymdownx.magiclink",
]
EXTENSION_CONFIGS = {
    "pymdownx.arithmatex": {"generic": True, "smart_dollar": False},
    "pymdownx.highlight": {"css_class": "highlight", "guess_lang": False},
    "footnotes": {"BACKLINK_TITLE": "Back to text"},
    "toc": {
        "permalink": True,
        "permalink_class": "anchor-link",
        "permalink_title": "Link to this heading",
        "slugify": slugify(case="lower"),
    },
}


def _converter() -> markdown.Markdown:
    return markdown.Markdown(extensions=EXTENSIONS, extension_configs=EXTENSION_CONFIGS)


# A pending mark (``mini.lit.document._Pending``) is HTML, which a code span or fence would
# show as text, and a highlighted fence would tokenise. So it crosses the converter as one
# plain word (hex keeps it alphanumeric, which every lexer keeps whole) and is a mark again
# in the output, wherever it landed.
_MARK_RE = re.compile(r'<mark class="pending">(.*?)</mark>')
_MARK_WORD_RE = re.compile(r"litpending([0-9a-f]+)z")


def _hide_marks(text: str) -> str:
    return _MARK_RE.sub(lambda m: f"litpending{m[1].encode().hex()}z", text)


def _show_marks(fragment: str) -> str:
    return _MARK_WORD_RE.sub(lambda m: f'<mark class="pending">{bytes.fromhex(m[1]).decode()}</mark>', fragment)


def to_html(text: str) -> str:
    """Render a Markdown document to an HTML fragment (a fresh converter per call, so footnote numbering starts at 1)."""
    return _show_marks(_converter().convert(_hide_marks(text)))


def render_fragment(text: str) -> str:
    """Render a short piece of Markdown (a caption) to HTML, with the same dialect as the document body."""
    return to_html(text)


@cache
def stylesheet() -> str:
    light = HtmlFormatter(style="default").get_style_defs(".highlight")
    dark = HtmlFormatter(style="github-dark").get_style_defs(".highlight")
    return f"{CSS_PATH.read_text()}\n{light}\n@media (prefers-color-scheme: dark) {{\n{dark}\n}}\n"


_KATEX = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {delimiters: [{left: '\\\\(', right: '\\\\)', display: false}, {left: '\\\\[', right: '\\\\]', display: true}]})"></script>
"""

_FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lora:wght@400..700&family=PT+Sans:wght@400;700&family=Fira+Mono:wght@400;500;700&display=swap">'


def page(body_html: str, *, title: str, extra_head: str = "", extra_body: str = "") -> str:
    """Wrap a rendered body in a complete, self-styled HTML page."""
    katex = _KATEX if 'class="arithmatex"' in body_html else ""
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"{_FONTS}\n<style>\n{stylesheet()}</style>\n{katex}{extra_head}</head>\n"
        f'<body>\n<main class="lit">\n{body_html}\n</main>\n{extra_body}</body>\n</html>\n'
    )


_TITLE_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.DOTALL)

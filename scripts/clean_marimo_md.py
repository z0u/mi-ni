#!/usr/bin/env python
"""Clean up ``marimo-md-export`` output into plain, LLM-friendly Markdown.

``marimo-md-export`` (see ``pyproject.toml``) already does the hard part of converting a notebook to Markdown with all string interpolation resolved. But cells that build their output programmatically (``mo.md(f"...")`` composed with ``mo.hstack``/callouts/tables, or run through the same renderer as ``mo.md``) come out as raw HTML fragments rather than Markdown, and figures are inlined as base64 PNGs. This script re-processes that output:

- drops the notebook frontmatter and ``<style>`` blocks
- externalizes inlined images (the light-theme variant only) to sibling files, referenced with normal ``![alt](...)`` Markdown
- converts ``<marimo-tex>`` spans to ``$...$`` / ``$$...$$``
- converts footnote refs/definitions to Markdown footnote syntax
- converts ``!!! type "Title"`` admonitions, and the ``<details>`` elements the same source produces from an interpolated cell, to GFM blockquotes
- converts the remaining HTML islands (``mo.md`` output, result tables) to Markdown via ``markdownify``, treating ``span.paragraph`` as a block
- keeps ``<figure>``/``<figcaption>`` tags, blank-line-separated so the Markdown inside them (image, caption) still renders as Markdown rather than being swallowed as raw HTML
- reflows each paragraph onto one line (``--no-reflow`` to keep the source wrapping), so a diff against an edited copy shows changed sentences rather than re-wrapping noise

Usage: ``uv run scripts/clean_marimo_md.py <in.md> [out.md]`` (defaults to overwriting in place). Images are written beside the output file, under ``<output-stem>.assets/`` unless ``--assets-dir`` is given.

To go from a notebook to a cleaned document in one step, use ``scripts/export_report_md.py``, which runs the export itself and then this pass. Prefer it over the ``marimo-md-export`` console script, which drops cells silently in a case that comes up in our reports — see that script's docstring.
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

from markdownify import MarkdownConverter

FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n+", re.DOTALL)
STYLE_RE = re.compile(r"<style>.*?</style>\n?", re.DOTALL)
EMPTY_COMMENT_RE = re.compile(r"^<!---->\n?", re.MULTILINE)

IMG_RE = re.compile(r"<img\b[^>]*>")
IMG_ATTR_RE = re.compile(r'([\w-]+)="([^"]*)"')
DATA_URI_RE = re.compile(r"data:(image/[\w.+-]+);base64,(.*)", re.DOTALL)
IMG_EXT_BY_MIME = {"image/png": "png", "image/svg+xml": "svg", "image/jpeg": "jpg"}

TEX_INLINE_RE = re.compile(r"<marimo-tex class=\"arithmatex\">\|\|\((.*?)\|\|\)</marimo-tex>", re.DOTALL)
TEX_BLOCK_RE = re.compile(r"<marimo-tex class=\"arithmatex\">\|\|\[(.*?)\|\|\]</marimo-tex>", re.DOTALL)
# NUL-delimited tokens so LaTeX content survives markdownify's escaping (which
# would mangle its backslashes/underscores/braces) untouched, and get swapped
# for real $...$ math only after escaping has already happened.
TEX_TOKEN_RE = re.compile("\x00(\\d+)\x00")

ADMONITION_RE = re.compile(
    r'^!!! (?P<type>\w+)(?: "(?P<title>[^"]*)")?\n(?P<body>(?:[ \t]{4}.*\n|[ \t]*\n)+)',
    re.MULTILINE,
)
ADMONITION_ICONS = {"note": "ℹ️ ", "tip": "💡 ", "important": "💬 ", "warning": "⚠️ ", "error": "🛑 "}

LEFTOVER_DIV_RE = re.compile(r"</?div\b[^>]*>\n?")

FOOTNOTE_REF_RE = re.compile(r'<sup id="fnref:([\w.-]+)"><a class="footnote-ref" href="#fn:\1">\d+</a></sup>')
FOOTNOTE_DIV_RE = re.compile(r'<div class="footnote">.*?</div>', re.DOTALL)
FOOTNOTE_LI_RE = re.compile(r'<li id="fn:([\w.-]+)">(.*?)</li>', re.DOTALL)
FOOTNOTE_BACKREF_RE = re.compile(r'\s*<a class="footnote-backref"[^>]*>.*?</a>\s*\Z', re.DOTALL)

FIGURE_OPEN_RE = re.compile(r"<figure\b[^>]*>\n?")
FIGURE_CLOSE_RE = re.compile(r"\n?(</figure>)")
FIGCAPTION_OPEN_RE = re.compile(r"(<figcaption>)\n?")
FIGCAPTION_CLOSE_RE = re.compile(r"\n?(</figcaption>)")


class ReportConverter(MarkdownConverter):
    """markdownify tweak: mo.md's paragraph spans are block-level, not inline."""

    def convert_span(self, el, text, parent_tags=None):
        classes = el.get("class") or []
        if "paragraph" in classes:
            return "\n\n" + text.strip() + "\n\n"
        return text

    def convert_details(self, el, text, parent_tags=None):
        # A `/// details | Title` aside reaches us two ways: as an `!!! details`
        # block when marimo unwrapped the cell to plain Markdown, and as
        # <details>/<summary> when it couldn't (an interpolated `mo.md(f"...")`
        # stays a code cell, so we get its rendered HTML instead). Both become
        # the same blockquote, so the same source reads the same either way.
        body = re.sub(r"\n{3,}", "\n\n", text.strip())
        return "\n\n" + "\n".join(f"> {ln}" if ln else ">" for ln in body.splitlines()) + "\n\n"

    def convert_summary(self, el, text, parent_tags=None):
        return f"**{text.strip()}**\n\n"

    def convert_sub(self, el, text, parent_tags=None):
        # No Markdown syntax for subscript/superscript; unlike the figure/
        # figcaption wrappers these are genuinely inline, so the raw tag
        # round-trips fine without any blank-line handling.
        return f"<sub>{text}</sub>"

    def convert_sup(self, el, text, parent_tags=None):
        return f"<sup>{text}</sup>"


def to_md(html: str) -> str:
    """Convert one HTML island to Markdown, shielding embedded LaTeX from escaping."""
    tex_by_token: dict[str, str] = {}

    def stash_tex(m: re.Match, is_block: bool) -> str:
        token = f"\x00{len(tex_by_token)}\x00"
        math = f"$${m.group(1)}$$" if is_block else f"${m.group(1)}$"
        # A '$' abutting a word character on either side reads as ambiguous
        # (e.g. currency) to most Markdown math extensions, so pad it.
        pre = m.string[m.start() - 1] if m.start() > 0 else ""
        post = m.string[m.end()] if m.end() < len(m.string) else ""
        lead = " " if pre.isalnum() else ""
        trail = " " if post.isalnum() else ""
        tex_by_token[token] = f"{lead}{math}{trail}"
        return token

    html = TEX_BLOCK_RE.sub(lambda m: stash_tex(m, True), html)
    html = TEX_INLINE_RE.sub(lambda m: stash_tex(m, False), html)

    out = ReportConverter(heading_style="ATX").convert(html).strip()
    return TEX_TOKEN_RE.sub(lambda m: tex_by_token[f"\x00{m.group(1)}\x00"], out)


def find_balanced(text: str, start: int, tag: str) -> int:
    """Return the index right after the ``</tag>`` matching the opening tag at ``start``."""
    open_re = re.compile(rf"<{tag}\b")
    close_re = re.compile(rf"</{tag}>")
    pos = text.index(">", start) + 1
    depth = 1
    while depth > 0:
        next_open = open_re.search(text, pos)
        next_close = close_re.search(text, pos)
        if next_close is None:
            raise ValueError(f"unbalanced <{tag}> starting at {start}")
        if next_open and next_open.start() < next_close.start():
            depth += 1
            pos = next_open.end()
        else:
            depth -= 1
            pos = next_close.end()
    return pos


def convert_balanced_islands(text: str, open_pattern: str, tag: str) -> str:
    """Replace every non-nested match of ``open_pattern`` (and its balanced close) with its Markdown."""
    out = []
    pos = 0
    open_re = re.compile(open_pattern)
    while True:
        m = open_re.search(text, pos)
        if not m:
            break
        out.append(text[pos : m.start()])
        end = find_balanced(text, m.start(), tag)
        inner = text[m.start() : end]
        out.append("\n\n" + to_md(inner) + "\n\n")
        pos = end
    out.append(text[pos:])
    return "".join(out)


def convert_footnote_defs(text: str) -> str:
    def replace_div(div_match: re.Match) -> str:
        lines = []
        for li in FOOTNOTE_LI_RE.finditer(div_match.group(0)):
            fn_id, body = li.group(1), li.group(2)
            body = FOOTNOTE_BACKREF_RE.sub("", body).strip()
            lines.append(f"[^{fn_id}]: {to_md(body)}")
        return "\n\n" + "\n\n".join(lines) + "\n\n"

    return FOOTNOTE_DIV_RE.sub(replace_div, text)


def convert_admonitions(text: str) -> str:
    def replace(m: re.Match) -> str:
        lines = m.group("body").splitlines()
        dedented = [re.sub(r"^[ \t]{1,4}", "", ln) if ln.strip() else "" for ln in lines]
        while dedented and dedented[-1] == "":
            dedented.pop()
        quoted = "\n".join(f"> {ln}" if ln else ">" for ln in dedented)

        title = m.group("title")
        header = ""
        if title and not re.fullmatch(r"<!--.*-->", title.strip()):
            icon = ADMONITION_ICONS.get(m.group("type"), "")
            header = f"> **{icon}{title}**\n>\n"
        return f"\n{header}{quoted}\n\n"

    return ADMONITION_RE.sub(replace, text)


def image_alt(raw: str) -> str:
    """An ``<img>``'s alt text, normalized for Markdown's ``![...]`` syntax.

    Alt text is written as prose and wraps across lines in the export; ``![...]`` is a single-line construct, and an unescaped ``]`` inside it would close the label early.
    """
    return re.sub(r"\s+", " ", raw).strip().replace("[", "(").replace("]", ")")


def externalize_images(text: str, assets_dir: Path, rel_dir: str) -> str:
    """Write the light-theme variant of each inlined image to ``assets_dir``, as ``![alt](...)``."""
    used_names: set[str] = set()

    def replace(m: re.Match) -> str:
        attrs = dict(IMG_ATTR_RE.findall(m.group(0)))
        classes = attrs.get("class", "").split()
        if "mini-themed-img-dark" in classes:
            return ""  # the light variant carries the same content; one copy is enough

        uri_match = DATA_URI_RE.match(attrs.get("src", ""))
        if not uri_match:
            return m.group(0)
        mime, b64data = uri_match.groups()
        ext = IMG_EXT_BY_MIME.get(mime, "bin")

        name = attrs.get("data-asset-name") or "fig"
        candidate = name
        n = 1
        while candidate in used_names:
            n += 1
            candidate = f"{name}-{n}"
        used_names.add(candidate)

        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / f"{candidate}.{ext}").write_bytes(base64.b64decode(b64data))

        return f"![{image_alt(attrs.get('alt', ''))}]({rel_dir}/{candidate}.{ext})"

    return IMG_RE.sub(replace, text)


def wrap_figures(text: str) -> str:
    # Once footnote/table islands are converted, any remaining <div> is just
    # layout wrapping (e.g. report-table-scroll's horizontal-scroll
    # container) with no content of its own left to preserve.
    text = LEFTOVER_DIV_RE.sub("", text)
    # CommonMark treats <figure>/<figcaption> as raw HTML blocks that swallow
    # everything up to the next blank line without reprocessing it as
    # Markdown. A blank line on both sides of each tag closes that HTML block
    # immediately, so the image/caption Markdown between tags renders normally.
    # The figure's class names (mini-themed-figure-*, report-figure) styled the
    # now-discarded <style> block; without it they're meaningless.
    text = FIGURE_OPEN_RE.sub("\n\n<figure>\n\n", text)
    text = FIGCAPTION_OPEN_RE.sub(r"\n\n\1\n\n", text)
    text = FIGCAPTION_CLOSE_RE.sub(r"\n\n\1\n\n", text)
    text = FIGURE_CLOSE_RE.sub(r"\n\n\1\n\n", text)
    return text


# A line that carries block structure of its own: joining it onto the previous
# line would either change what it means or destroy the block it belongs to.
STRUCTURAL_RE = re.compile(
    r"""^(?:
          [ \t]{4,}\S              # indented code
        | \s*(?:[-*+]|\d+[.)])\s   # list item
        | \s*\#{1,6}\s             # ATX heading
        | \s*>                     # blockquote
        | \s*\|                    # table row
        | \s*<                     # HTML island (figure, figcaption, sub/sup)
        | \s*\$\$                  # display math
        | \s*\[\^[\w.-]+\]:        # footnote definition
        | \s*(?:-{3,}|={3,}|\*{3,})\s*$  # setext underline or thematic break
    )""",
    re.VERBOSE,
)
FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
# A trailing backslash is a hard line break, so the join has to stop there. The
# two-space spelling never reaches this point: the whitespace pass below strips
# it (and marimo's renderer wouldn't have produced one anyway).
HARD_BREAK_RE = re.compile(r"\\$")


def reflow(text: str) -> str:
    """Join each paragraph's soft-wrapped lines into one line."""
    out: list[str] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        if buf:
            out.append(" ".join(ln.strip() for ln in buf))
            buf.clear()

    for line in text.splitlines():
        if FENCE_RE.match(line):
            flush()
            in_fence = not in_fence
            out.append(line)
        elif in_fence or not line.strip() or STRUCTURAL_RE.match(line):
            flush()
            out.append(line)
        elif HARD_BREAK_RE.search(line):
            buf.append(line)
            flush()
        else:
            buf.append(line)
    flush()
    return "\n".join(out)


def clean(text: str, assets_dir: Path, rel_dir: str, do_reflow: bool = True) -> str:
    text = FRONTMATTER_RE.sub("", text)
    text = STYLE_RE.sub("", text)
    text = EMPTY_COMMENT_RE.sub("", text)
    text = externalize_images(text, assets_dir, rel_dir)
    text = FOOTNOTE_REF_RE.sub(lambda m: f"[^{m.group(1)}]", text)
    text = convert_footnote_defs(text)
    text = convert_admonitions(text)
    text = convert_balanced_islands(text, r'<span class="markdown prose dark:prose-invert contents">', "span")
    text = convert_balanced_islands(text, r'<table class="report-table">', "table")
    text = wrap_figures(text)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if do_reflow:
        text = reflow(text)
    return text.strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?", help="defaults to overwriting the input")
    parser.add_argument("--assets-dir", type=Path, help="defaults to <output-stem>.assets/ beside the output")
    parser.add_argument("--no-reflow", action="store_true", help="keep the source's line wrapping")
    args = parser.parse_args()

    dst = args.output or args.input
    assets_dir = args.assets_dir or dst.with_name(f"{dst.stem}.assets")
    rel_dir = assets_dir.name if assets_dir.parent == dst.parent else str(assets_dir)

    dst.write_text(clean(args.input.read_text(), assets_dir, rel_dir, do_reflow=not args.no_reflow))


if __name__ == "__main__":
    main()

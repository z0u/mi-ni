"""Paragraph reflow and HTML-island conversion in the marimo-export cleaner."""

import re

from tests.conftest import load_script

clean_marimo_md = load_script("clean_marimo_md")

reflow = clean_marimo_md.reflow
to_md = clean_marimo_md.to_md
convert_admonitions = clean_marimo_md.convert_admonitions


def test_joins_soft_wrapped_paragraphs():
    assert reflow("one two\nthree four\n\nnext para\nhere") == "one two three four\n\nnext para here"


def test_leaves_structure_alone():
    # Each of these carries block meaning that a join would change or destroy.
    for block in (
        "- item one\n- item two",
        "1. first\n2. second",
        "| a | b |\n| --- | --- |",
        "# Heading\n## Sub",
        "> quoted\n> more",
        "    indented code\n    more code",
        "<figure>\n<figcaption>",
        "[^note]: a definition",
        "$$\nx = 1\n$$",
    ):
        assert reflow(block) == block, block


def test_leaves_fenced_code_alone():
    src = "prose here\nwrapped\n\n```python\nx = 1\n\ny = 2\n```\n\nmore\nprose"
    assert reflow(src) == "prose here wrapped\n\n```python\nx = 1\n\ny = 2\n```\n\nmore prose"


def test_structural_line_breaks_the_join():
    # A list may follow a paragraph with no blank line between them.
    assert reflow("lead in\ncontinues\n- item\n- item") == "lead in continues\n- item\n- item"


def test_keeps_hard_line_breaks():
    assert reflow("line one\\\nline two") == "line one\\\nline two"


def test_details_and_admonition_agree():
    # A `/// details | Title` aside arrives as an `!!! details` block when marimo
    # unwrapped the cell to markdown, and as <details>/<summary> when it stayed a
    # code cell (an interpolated `mo.md(f"...")`). Same source, same rendering.
    from_html = to_md("<details><summary>Title</summary><span class='paragraph'>Body.</span></details>")
    from_markdown = convert_admonitions('!!! details "Title"\n    Body.\n').strip()
    assert from_html == from_markdown == "> **Title**\n>\n> Body."


def test_reflow_preserves_content():
    """Every paragraph is joined and no word is lost, with the blocks between them left where they were."""
    src = "some prose\nwrapped oddly\n\n- a list\n\n| t | b |\n\ntrailing\ntext\n"
    out = reflow(src)
    squash = lambda t: re.sub(r"\s+", " ", t).strip()  # noqa: E731
    assert squash(out) == squash(src)
    assert out.splitlines() == ["some prose wrapped oddly", "", "- a list", "", "| t | b |", "", "trailing text"]

"""Paragraph reflow, HTML-island conversion, and sidecar linking in the marimo-export cleaner."""

import re

from mini.reports import Publisher, externalize_html

from tests.conftest import load_script

clean_marimo_md = load_script("clean_marimo_md")

reflow = clean_marimo_md.reflow
to_md = clean_marimo_md.to_md
convert_admonitions = clean_marimo_md.convert_admonitions
link_externalized = clean_marimo_md.link_externalized


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


def test_a_stamped_fragment_becomes_a_link_to_its_sidecar():
    """The point of the pass: a screenful of path data leaves the document, its description stays."""
    frag = (
        '<figure data-mini-asset="public/.mini/report/sublines.html?v=1a2b3c4d" aria-label="Two sublines.">'
        '<figure><svg viewBox="0 0 9 9"><path d="M0 0 L9 9"/></svg><figcaption>anchored</figcaption></figure>'
        "</figure>"
    )
    assert link_externalized(f"before{frag}after") == (
        "before\n\n[Two sublines.](public/.mini/report/sublines.html?v=1a2b3c4d)\n\nafter"
    )


def test_the_whole_fragment_goes_even_when_it_nests_its_own_tags():
    """Nested <figure>s (a captioned sub-figure per condition) must not end the element early."""
    frag = '<figure data-mini-asset="_assets/s.html" aria-label="X"><figure>a</figure><figure>b</figure></figure>'
    assert link_externalized(f"{frag}tail") == "\n\n[X](_assets/s.html)\n\ntail"


def test_an_svg_sidecar_is_written_as_an_image():
    """`.html` is a document to follow; `.svg` is a figure a reader's viewer can show in place."""
    frag = '<svg data-mini-asset="_assets/spark.svg" aria-label="A spark."><path d="M0 0"/></svg>'
    assert link_externalized(frag).strip() == "![A spark.](_assets/spark.svg)"


def test_an_undescribed_fragment_falls_back_to_the_sidecars_name(capsys):
    frag = '<figure data-mini-asset="_assets/mystery.html">…</figure>'
    assert link_externalized(frag).strip() == "[mystery](_assets/mystery.html)"
    assert "aria-label" in capsys.readouterr().err  # and the author hears about it


def test_unstamped_markup_is_left_for_the_passes_below():
    """Only a fragment with a sidecar can be swapped for a link — everything else still converts."""
    md = '<figure><img src="x.png" alt="A plot."></figure>'
    assert link_externalized(md) == md


def test_the_stamp_the_library_writes_is_the_one_this_reads(tmp_path):
    """The two ends agree on the marker — the pass is driven by what `externalize_html` produces."""
    inline = externalize_html(
        '<figure aria-label="A [bracketed] label.">'
        '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg></figure>',
        name="strip",
        publish=Publisher(tmp_path / "_assets"),
    )
    # Brackets in the label would close the link text early, so they are normalized.
    assert link_externalized(inline).strip() == "[A (bracketed) label.](_assets/strip.html)"

"""Fence handling, the dropped-cell guard, and figure-link localization in the report exporter."""

from pathlib import Path

import pytest
from marimo_md_export.models import Cell

from tests.conftest import load_script

export_report_md = load_script("export_report_md")

convert_admonitions = export_report_md.convert_admonitions
fenced_spans = export_report_md.fenced_spans
check_sources_agree = export_report_md.check_sources_agree
localize_links = export_report_md.localize_links

# One code cell, holding a `///` admonition inside the Python source. This is
# what an interpolated `mo.md(f"...")` looks like in the markdown export: marimo
# can only unwrap literal `mo.md("...")` cells to plain markdown, so this one
# stays a fence and its source has to survive verbatim to match its output.
CELL_MD = '```python {.marimo hide_code="true"}\nmo.md(rf"""\n/// details | Aside\nBody.\n///\n""")\n```\n'


def test_admonition_outside_code_is_converted():
    assert convert_admonitions("/// tip | Title\nBody.\n///\n") == '!!! tip "Title"\n    Body.\n'


def test_admonition_inside_code_is_left_alone():
    # The bug this guards: rewriting the fence changes the cell's source hash,
    # so its rendered output no longer matches and the cell is dropped.
    assert convert_admonitions(CELL_MD) == CELL_MD


def test_admonitions_around_a_fence_are_still_converted():
    src = f"/// tip | Before\nB.\n///\n\n{CELL_MD}\n/// tip | After\nA.\n///\n"
    out = convert_admonitions(src)
    assert '!!! tip "Before"' in out
    assert '!!! tip "After"' in out
    assert CELL_MD in out


def test_fenced_spans_covers_the_whole_block():
    src = f"lead\n\n{CELL_MD}trail\n"
    ((start, end),) = fenced_spans(src)
    assert src[start:end] == CELL_MD


def test_backticks_inside_a_widened_fence_do_not_close_it():
    # marimo widens a cell's fence when the source contains backticks.
    src = '````python {.marimo}\nmo.md("```\\nx\\n```")\n````\n'
    ((start, end),) = fenced_spans(src)
    assert src[start:end] == src


def test_closing_fence_may_not_carry_an_info_string():
    src = "```\na\n```python\nb\n```\n"
    assert [src[s:e] for s, e in fenced_spans(src)] == ["```\na\n```python\nb\n```\n"]


def _cell(source: str) -> Cell:
    from marimo_md_export.parse_md import _md5

    return Cell(source=source, source_hash=_md5(source.strip()), block_text=source)


def test_check_raises_on_a_rewritten_fence():
    cell = _cell("x = 1\n")
    check_sources_agree([cell], {cell.source_hash})  # the control: an untouched source agrees
    with pytest.raises(RuntimeError, match="no notebook cell has"):
        check_sources_agree([cell], {"some-other-hash"})


# --- figure links ---------------------------------------------------------
# A render lands under .mini/renders/, two levels away from the notebook whose
# public/.mini/ holds the figures, so every link the export wrote has to move.


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """A notebook dir holding one published figure, and the render's dir elsewhere in the tree."""
    (nb_dir := tmp_path / "docs" / "ex-1").mkdir(parents=True)
    (figs := nb_dir / "public" / ".mini" / "report").mkdir(parents=True)
    (figs / "cube-light.png").write_bytes(b"\x89PNG")
    (out_dir := tmp_path / ".mini" / "renders").mkdir(parents=True)
    return nb_dir, out_dir


def test_img_tag_becomes_a_markdown_image_pointing_at_the_file(dirs):
    nb_dir, out_dir = dirs
    md = '<img src="public/.mini/report/cube-light.png?v=72ff1adc" alt="A cube,\nrotated." width="434" />'
    out, unresolved = localize_links(md, base=nb_dir, out_dir=out_dir)
    assert out == "![A cube, rotated.](../../docs/ex-1/public/.mini/report/cube-light.png)"
    assert unresolved == []


def test_a_markdown_link_is_repointed_too(dirs):
    nb_dir, out_dir = dirs
    out, _ = localize_links("![c](public/.mini/report/cube-light.png)", base=nb_dir, out_dir=out_dir)
    assert out == "![c](../../docs/ex-1/public/.mini/report/cube-light.png)"


def test_an_image_written_beside_the_output_is_left_alone(dirs):
    """`clean_marimo_md` externalizes inlined images into `<stem>.assets/`, already relative to the output."""
    _, out_dir = dirs
    (assets := out_dir / "ex-1.assets").mkdir()
    (assets / "plot.png").write_bytes(b"\x89PNG")
    md = "![p](ex-1.assets/plot.png)"
    assert localize_links(md, base=dirs[0], out_dir=out_dir) == (md, [])


@pytest.mark.parametrize("src", ["https://example.test/x.png", "data:image/png;base64,iVBOR"])
def test_a_link_that_names_no_file_at_all_passes_through_silently(dirs, src):
    """An off-repo URL and an inlined image are complete as they stand — nothing to repoint, nothing missing."""
    nb_dir, out_dir = dirs
    md = f"![x]({src})"
    assert localize_links(md, base=nb_dir, out_dir=out_dir) == (md, [])


def test_a_local_link_that_resolves_nowhere_is_reported(dirs):
    """The link stays (a broken link reads better than a dropped figure), but the caller gets to warn."""
    nb_dir, out_dir = dirs
    md = "![x](public/.mini/report/gone.png)"
    assert localize_links(md, base=nb_dir, out_dir=out_dir) == (md, ["public/.mini/report/gone.png"])

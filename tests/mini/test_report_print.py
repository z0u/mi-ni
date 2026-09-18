"""The export's PDF print: stable bytes, and a quiet exit without a browser."""

from pathlib import Path

import pikepdf
import pytest

from mini import report_print

_PAGE = "<html><body><h1>Hi</h1><p>A page with <a href='https://example.test/'>a link</a>.</p></body></html>"


@pytest.fixture
def browser():
    """A headless Chromium, or skip: the CI runner has none, and that is the case the exporter's fallback covers. Function-scoped, since Playwright's sync API owns one event loop per context and the next test opens its own."""
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=report_print.chromium_path())
        except Error as e:
            pytest.skip(f"no Chromium: {e}")
        yield browser
        browser.close()


def test_two_prints_of_one_page_are_byte_equal(browser, tmp_path: Path):
    """An unchanged report must re-publish as an unchanged bundle; Chromium's dates and random ID would break that."""
    outs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for out in outs:
        page = browser.new_page()
        page.set_content(_PAGE)
        page.wait_for_timeout(1100)  # CreationDate has one-second resolution
        report_print.print_page(page, out, settle=0)
        page.close()
    assert outs[0].read_bytes() == outs[1].read_bytes()
    with pikepdf.open(outs[0]) as pdf:
        assert {"/CreationDate", "/ModDate"}.isdisjoint(pdf.docinfo.keys())


def test_print_bundle_skips_without_a_browser(tmp_path: Path, monkeypatch, caplog):
    """A publish must not fail on the PDF: no Chromium means no file, one warning, and the bundle untouched."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "index.html").write_text(_PAGE)
    monkeypatch.setattr(report_print, "chromium_path", lambda: str(tmp_path / "no-such-chromium"))
    out = report_print.print_bundle(bundle, bundle / "report.pdf")
    assert out is None
    assert not (bundle / "report.pdf").exists()
    assert not (tmp_path / ".render-bundle").exists()  # the serve root is cleaned up on the way out
    assert [r.message[:29] for r in caplog.records] == ["report PDF skipped: no Chromi"]


_SECTIONED = (
    """<html><head><style>
@page { size: 100mm 120mm; margin: 10mm }
h2 { break-before: page }
</style></head><body>
<h1>Title</h1><p>A short first page.</p>
<h2>Long</h2>"""
    + "<p>Line.</p>" * 60
    + """
<h2>Short</h2><p>One line.</p>
</body></html>"""
)

_MM = 72 / 25.4  # points per mm


def test_fit_prints_one_page_per_section_and_clips_each_to_its_ink(browser, tmp_path: Path):
    """The long section overflows the stylesheet's page; the print grows the sheet until it fits, then cuts every page to what is on it."""
    out = tmp_path / "fit.pdf"
    page = browser.new_page()
    page.set_content(_SECTIONED)
    report_print.print_page(page, out, settle=0)
    page.close()
    with pikepdf.open(out) as pdf:
        heights = [float(p.MediaBox[3]) - float(p.MediaBox[1]) for p in pdf.pages]
    assert len(heights) == 3  # title page + two sections, none broken across pages
    assert heights[1] > 120 * _MM  # the long section needed more than the stylesheet's page
    assert heights[0] < 60 * _MM and heights[2] < 60 * _MM  # the short ones were cut back to their content


def test_fit_leaves_a_page_without_a_sized_page_rule_alone(browser, tmp_path: Path):
    out = tmp_path / "plain.pdf"
    page = browser.new_page()
    page.set_content(_PAGE)
    report_print.print_page(page, out, settle=0)
    page.close()
    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 1
        assert float(pdf.pages[0].MediaBox[3]) == pytest.approx(11 * 72, abs=1)  # Chromium's default letter page, uncut


def test_ink_extents_of_a_blank_page_is_none(tmp_path: Path):
    out = tmp_path / "blank.pdf"
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 400))
    pdf.save(out)
    assert report_print.ink_extents(out) == [None]

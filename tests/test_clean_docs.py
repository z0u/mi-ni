"""Post-export HTML tidy-ups: the code-collapsed-by-default shim."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "clean_docs", Path(__file__).resolve().parent.parent / "scripts" / "clean_docs.py"
)
assert _SPEC and _SPEC.loader
clean_docs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(clean_docs)


_EXPORT = '<!doctype html><html><head><meta charset="utf-8"></head><body>…</body></html>'


def test_injects_shim_into_head(tmp_path):
    p = tmp_path / "index.html"
    p.write_text(_EXPORT, "utf-8")
    assert clean_docs.default_hidden_code(p) is True
    out = p.read_text("utf-8")
    # Shim lands inside <head>, before any body content, so it runs before marimo's bundle.
    assert clean_docs._HIDE_CODE_MARKER in out
    assert out.index("</head>") > out.index(clean_docs._HIDE_CODE_MARKER)
    # It seeds the show-code param without clobbering an explicit one.
    assert 'set("show-code","false")' in out
    assert 'has("show-code")' in out


def test_idempotent(tmp_path):
    p = tmp_path / "index.html"
    p.write_text(_EXPORT, "utf-8")
    clean_docs.default_hidden_code(p)
    once = p.read_text("utf-8")
    assert clean_docs.default_hidden_code(p) is False  # marker already present — no-op
    assert p.read_text("utf-8") == once  # and not injected twice


def test_noop_when_no_head(tmp_path):
    """A document without a <head> (or an unexpected format) is left untouched."""
    p = tmp_path / "index.html"
    p.write_text("<div>no head here</div>", "utf-8")
    assert clean_docs.default_hidden_code(p) is False

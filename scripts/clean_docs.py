#!/usr/bin/env python
"""Post-export tidy-ups for Marimo HTML and session JSON: collapse terminal control
sequences, redact URLs that shouldn't ship, and hide report code by default.
"""

import json
import re
import sys
from pathlib import Path

# Exported report bundles (gitignored). clean_html/clean_session_json are also imported
# by export_reports.py and applied at export time, before a bundle is synced.
EXPORTS_DIR = Path(__file__).parent.parent / ".mini" / "exports"

_CSI = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")

REDACT: list[tuple[re.Pattern, str]] = [
    (re.compile(r"https://modal\.com/apps/\S+"), "[modal.com/apps/…]"),
]


def _needs_redaction(text: str) -> bool:
    return any(p.search(text) for p, _ in REDACT)


def _redact(text: str) -> str:
    """Redact patterns that should not appear in published output."""
    for pattern, replacement in REDACT:
        text = pattern.sub(replacement, text)
    return text


def _apply_terminal(text: str) -> str:  # noqa: C901
    """Collapse CR/erase/cursor-up sequences; keep SGR color codes."""
    lines: list[list[str]] = [[]]
    row = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\n":
            row += 1
            while len(lines) <= row:
                lines.append([])
            i += 1
        elif c == "\r":
            lines[row] = []
            i += 1
        elif c == "\x1b":
            m = _CSI.match(text, i)
            if m:
                params, cmd = m.group(1), m.group(2)
                if cmd == "m":
                    lines[row].append(m.group(0))  # SGR: keep color/style
                elif cmd == "K":
                    lines[row] = []  # erase line (any variant)
                elif cmd == "A":
                    n = int(params) if params and params.isdigit() else 1
                    row = max(0, row - n)
                elif cmd == "J" and params == "2":
                    lines, row = [[]], 0
                # all other CSI (cursor pos, mode, hide/show cursor) — drop
                i = m.end()
            else:
                i += 1  # bare or unrecognised ESC — skip
        else:
            lines[row].append(c)
            i += 1

    result = "\n".join("".join(line) for line in lines).strip()
    return re.sub(r"\n{3,}", "\n\n", result)


# --- HTML cleaning (regex-based, avoids parsing the JS wrapper) ---

_TEXT_FIELD = re.compile(r'("text":\s*")((?:[^"\\]|\\.)*)"')


def _has_control_seqs(raw: str) -> bool:
    return "\\r" in raw or "\\u001b" in raw


def clean_html(path: Path) -> bool:
    content = path.read_text("utf-8")

    def replace(m: re.Match) -> str:
        prefix, inner = m.group(1), m.group(2)
        if not _has_control_seqs(inner) and not _needs_redaction(inner):
            return m.group(0)
        try:
            text = json.loads(f'"{inner}"')
        except json.JSONDecodeError:
            return m.group(0)
        cleaned = _apply_terminal(text) if _has_control_seqs(inner) else text
        cleaned = _redact(cleaned)
        if cleaned == text:
            return m.group(0)
        return prefix + json.dumps(cleaned)[1:-1] + '"'

    new_content = _TEXT_FIELD.sub(replace, content)
    if new_content == content:
        return False
    path.write_text(new_content, "utf-8")
    return True


# --- Default to hidden code ---
#
# Our reports are literate (narrated prose and figures), so they read better with the
# code out of the way. marimo's read view already ships a "Show code" toggle in its
# menu; it just starts *on*. The obvious lever — flipping the `showAppCode` app-config
# to false — is wrong: `canShowCode` short-circuits on that same flag, so false removes
# the whole menu (toggle included), leaving the source reachable only via the "download
# code" link. Instead marimo seeds the toggle's initial state from a `?show-code` query
# param (falling back to `showAppCode` when absent). So we keep the config's default and
# inject a tiny head script that sets `?show-code=false` when no such param is present:
# code starts collapsed, the toggle stays, and a deliberate `?show-code=true` link still
# opens with code shown. The script runs during head parsing, before marimo's (module)
# bundle reads the URL. try/catch means any failure degrades to marimo's default rather
# than breaking the page.

_HIDE_CODE_MARKER = "mini:default-hidden-code"
_HIDE_CODE_SHIM = (
    f"<script>/* {_HIDE_CODE_MARKER} */(function(){{try{{"
    "var u=new URL(window.location.href);"
    'if(!u.searchParams.has("show-code")){'
    'u.searchParams.set("show-code","false");'
    'window.history.replaceState(null,"",u.toString());'
    "}}catch(e){}})();</script>"
)
_HEAD_OPEN = re.compile(r"<head\b[^>]*>", re.IGNORECASE)


def default_hidden_code(path: Path) -> bool:
    """Inject the code-collapsed-by-default shim into a report's <head> (idempotent)."""
    content = path.read_text("utf-8")
    if _HIDE_CODE_MARKER in content:
        return False
    new_content, n = _HEAD_OPEN.subn(lambda m: m.group(0) + _HIDE_CODE_SHIM, content, count=1)
    if not n:
        return False
    path.write_text(new_content, "utf-8")
    return True


# --- Session JSON cleaning (proper JSON parse/dump) ---


def _clean_console(entries: list) -> bool:
    """Clean console entries in-place. Returns True if any changed."""
    changed = False
    for entry in entries:
        text = entry.get("text", "")
        if not text:
            continue
        needs_clean = "\r" in text or "\x1b" in text
        if not needs_clean and not _needs_redaction(text):
            continue
        cleaned = _apply_terminal(text) if needs_clean else text
        cleaned = _redact(cleaned)
        if cleaned != text:
            entry["text"] = cleaned
            changed = True
    return changed


def clean_session_json(path: Path) -> bool:
    data = json.loads(path.read_text("utf-8"))
    changed = False
    for cell in data.get("cells", []):
        if _clean_console(cell.get("console", [])):
            changed = True
    if not changed:
        return False
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return True


# --- Main ---


def _clean(path: Path) -> bool:
    if path.suffix == ".html":
        # `|` not `or`: run both, don't short-circuit on the first that changes nothing.
        return clean_html(path) | default_hidden_code(path)
    if path.name.endswith(".py.json"):
        return clean_session_json(path)
    return False


def main() -> None:
    if sys.argv[1:]:
        paths = [Path(a) for a in sys.argv[1:]]
    else:
        paths = list(EXPORTS_DIR.rglob("*.html")) + list(EXPORTS_DIR.rglob("*.py.json"))

    changed = 0
    for p in paths:
        if _clean(p):
            print(f"cleaned: {p}")
            changed += 1
        else:
            print(f"unchanged: {p}")
    print(f"{changed}/{len(paths)} files updated")


if __name__ == "__main__":
    main()

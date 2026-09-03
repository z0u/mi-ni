#!/usr/bin/env python
"""Run ``marimo edit`` on a notebook, and print a URL that opens straight into the app view.

The loop we actually work in is: edit the source in the IDE, watch the rendering in a browser tab. ``--watch`` is what makes that work — marimo re-runs on save, so the IDE stays the editor — and ``--headless`` stops marimo grabbing an external browser, leaving a URL that Cmd-click opens in VS Code's own one. That is a lot to type, and the tab still lands in the code view; reaching the rendering means finding the app-view toggle with the mouse.

marimo's frontend reads a ``view-as=present`` query parameter in edit mode, so the app view can be a property of the link instead of a click. We let marimo choose its own port and token, then rewrite the URL banner it prints: one link in, two out, app view first.

Rewriting the banner rather than printing our own URL is what keeps a single link on screen at the moment you want to click one. It costs a pipe on marimo's stdout, and a pipe is where a short banner would otherwise sit in a block buffer indefinitely — hence ``PYTHONUNBUFFERED`` below.
"""

import os
import re
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit

ANSI = re.compile(r"\x1b\[[0-9;]*m")
BANNER = re.compile(r"^(?P<lead>\s*(?:➜\s+)?)URL:\s+(?P<url>https?://\S+)\s*$")


def app_view(url: str) -> str:
    """Point *url* at marimo's app view, keeping whatever query it already carries (the access token)."""
    parts = urlsplit(url)
    query = f"{parts.query}&view-as=present" if parts.query else "view-as=present"
    return urlunsplit(parts._replace(query=query))


def rewrite(line: str) -> str | None:
    """Expand marimo's ``URL:`` banner into an app-view link and the editor one, or ``None`` if *line* is anything else.

    Matched against an ANSI-stripped copy so a coloured banner still parses; the replacement is plain text we build ourselves, so nothing has to be put back.
    """
    match = BANNER.match(ANSI.sub("", line))
    if match is None:
        return None
    lead, url = match["lead"], match["url"]
    return f"{lead}App:    {app_view(url)}\n{lead}Editor: {url}"


def marimo_command(args: list[str]) -> list[str]:
    """Build the ``marimo edit`` invocation, consuming our own ``--browser`` and passing the rest through.

    ``--browser`` hands the opening back to marimo (and to whatever ``$BROWSER`` is), for anyone not driving this from an editor terminal. It opens the editor URL rather than ours, since marimo builds that one itself.
    """
    browser = "--browser" in args
    passthrough = [a for a in args if a != "--browser"]
    return [sys.executable, "-m", "marimo", "edit", "--watch", *([] if browser else ["--headless"]), *passthrough]


def exit_status(returncode: int) -> int:
    """Translate ``Popen.wait()`` into a shell exit status.

    A signalled child reports a negative number, which ``SystemExit`` then wraps into whatever the low byte happens to be — Ctrl-C would surface as 241 rather than the 130 a shell expects. The usual ``128 + signal`` mapping keeps it legible to the caller.
    """
    return 128 - returncode if returncode < 0 else returncode


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(f"usage: {sys.argv[0]} [--browser] <notebook.py> [marimo edit options]", file=sys.stderr)
        return 2

    cmd = marimo_command(args)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, text=True, bufsize=1, env=os.environ | {"PYTHONUNBUFFERED": "1"}
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(rewrite(line) or line.rstrip("\n"), flush=True)
    except KeyboardInterrupt:
        # Ctrl-C reached marimo too (same process group), so it is already shutting
        # down; just stop reading and collect its status.
        pass
    return exit_status(proc.wait())


if __name__ == "__main__":
    raise SystemExit(main())

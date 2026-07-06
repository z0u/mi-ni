#!/usr/bin/env python
"""Run a command, intercept auth URLs in its output, and render them as QR codes or open in browser."""

import errno
import os
import pty
import re
import select
import sys
import webbrowser
from urllib.parse import urlparse

import segno

URL_RE = re.compile(rb'https?://[^\s\x1b\'"`<>]+')
AUTH_PATH_RE = re.compile(r"/(auth|authorize|login|setup|token|key|verify|confirm|activate)", re.IGNORECASE)


def is_auth_url(url: str) -> bool:
    return bool(AUTH_PATH_RE.search(urlparse(url).path))


def render_qr(url: str) -> None:
    qr = segno.make(url, error="l")
    # Clear the current line because some tools (e.g. Modal) use a spinner that
    # overwrites the same line repeatedly.
    print("\033[2K\r", end="", flush=True, file=sys.stderr)
    qr.terminal(out=sys.stdout, compact=True)
    sys.stdout.flush()


def open_url(url: str) -> None:
    print("\033[2K\r", end="", flush=True, file=sys.stderr)
    if not webbrowser.open(url):
        render_qr(url)


def scan_lines(buf: bytearray, seen: set[bytes], handle_url) -> None:
    while True:
        nl = buf.find(b"\n")
        if nl < 0:
            return
        line = bytes(buf[:nl])
        del buf[: nl + 1]
        for match in URL_RE.findall(line):
            if match in seen:
                continue
            seen.add(match)
            url = match.decode("utf-8", "replace")
            if not is_auth_url(url):
                continue
            try:
                handle_url(url)
            except Exception as e:
                sys.stderr.write(f"(URL handling failed: {e})\n")


def pump(child_fd: int, in_fd: int, out_fd: int, handle_url) -> None:
    seen: set[bytes] = set()
    buf = bytearray()
    while True:
        try:
            rlist, _, _ = select.select([child_fd, in_fd], [], [])
        except InterruptedError:
            continue
        if in_fd in rlist:
            try:
                data = os.read(in_fd, 4096)
            except OSError:
                data = b""
            if data:
                os.write(child_fd, data)
        if child_fd in rlist:
            try:
                chunk = os.read(child_fd, 4096)
            except OSError as e:
                if e.errno == errno.EIO:
                    return
                raise
            if not chunk:
                return
            os.write(out_fd, chunk)
            buf.extend(chunk)
            scan_lines(buf, seen, handle_url)


def main() -> int:
    args = sys.argv[1:]

    mode = "open"
    if args and args[0] in ("--qr", "--open"):
        mode = args[0][2:]
        args = args[1:]

    if not args:
        sys.stderr.write(f"Usage: {sys.argv[0]} [--qr|--open] <cmd...>\n")
        return 1

    handle_url = render_qr if mode == "qr" else open_url

    # Open/render a literal URL (e.g. one we constructed) rather than a command's output.
    if args and args[0] == "--url":
        if len(args) < 2:
            sys.stderr.write(f"Usage: {sys.argv[0]} [--qr|--open] --url <URL>\n")
            return 1
        handle_url(args[1])
        return 0

    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(args[0], args)

    try:
        pump(fd, sys.stdin.fileno(), sys.stdout.fileno(), handle_url)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


if __name__ == "__main__":
    sys.exit(main())

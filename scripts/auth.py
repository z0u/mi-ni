#!/usr/bin/env python
"""Run `modal setup` and render any auth URLs in its output as QR codes."""

import errno
import os
import pty
import re
import select
import sys
import termios
import tty

import segno

URL_RE = re.compile(rb'https?://[^\s\x1b\'"`<>]+')


def render_qr(url: str) -> None:
    qr = segno.make(url, error='l')
    qr.terminal(out=sys.stdout, compact=True)
    sys.stdout.flush()


def scan_lines(buf: bytearray, seen: set[bytes]) -> None:
    while True:
        nl = buf.find(b'\n')
        if nl < 0:
            return
        line = bytes(buf[:nl])
        del buf[: nl + 1]
        for match in URL_RE.findall(line):
            if match in seen:
                continue
            seen.add(match)
            try:
                render_qr(match.decode('utf-8', 'replace'))
            except Exception as e:
                sys.stderr.write(f'(QR render failed: {e})\n')


def pump(child_fd: int, in_fd: int, out_fd: int) -> None:
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
                data = b''
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
            scan_lines(buf, seen)


def main() -> int:
    cmd = ['uv', 'run', 'modal', 'setup', *sys.argv[1:]]

    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(cmd[0], cmd)

    in_fd = sys.stdin.fileno()
    old_attrs = None
    if os.isatty(in_fd):
        old_attrs = termios.tcgetattr(in_fd)
        tty.setraw(in_fd)

    try:
        pump(fd, in_fd, sys.stdout.fileno())
    finally:
        if old_attrs is not None:
            termios.tcsetattr(in_fd, termios.TCSADRAIN, old_attrs)
        try:
            os.close(fd)
        except OSError:
            pass

    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


if __name__ == '__main__':
    sys.exit(main())

"""Tests for the RichProgressDisplay logging integration."""

from __future__ import annotations

import io
import logging
import time

from rich.console import Console
from rich.logging import RichHandler

from mini._queues import EndOfQueue
from mini.local_queue import LocalQueue
from mini.progress import ProgressMessage
from mini.progress_display import RichProgressDisplay, _route_logging_to


def test_route_logging_to_swaps_and_restores_root_handlers():
    """Root handlers are replaced inside the context and restored on exit."""
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    saved = root.handlers[:]
    root.handlers = [sentinel]
    try:
        console = Console(file=io.StringIO(), force_terminal=False)
        with _route_logging_to(console):
            assert root.handlers != [sentinel]
            assert len(root.handlers) == 1
            assert isinstance(root.handlers[0], RichHandler)
        assert root.handlers == [sentinel]
    finally:
        root.handlers = saved


def test_route_logging_to_writes_records_through_console():
    """Log records emitted inside the context land in the supplied console."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=120)

    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        root.setLevel(logging.INFO)
        with _route_logging_to(console):
            logging.getLogger("mini.test").info("hello-from-test")
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)

    output = buf.getvalue()
    assert "hello-from-test" in output


def test_route_logging_to_restores_on_exception():
    """Handlers are restored even if the body raises."""
    root = logging.getLogger()
    saved = root.handlers[:]
    marker = logging.NullHandler()
    root.handlers = [marker]
    try:
        console = Console(file=io.StringIO(), force_terminal=False)
        try:
            with _route_logging_to(console):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert root.handlers == [marker]
    finally:
        root.handlers = saved


def test_rich_progress_display_routes_logging_while_running():
    """While the live display is running, logging goes through its console."""
    buf = io.StringIO()
    queue: LocalQueue[ProgressMessage] = LocalQueue()
    display = RichProgressDisplay(total_jobs=1, queue=queue)
    display.console = Console(file=buf, force_terminal=False, width=120)

    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        root.setLevel(logging.WARNING)
        with display:
            # Wait until the display thread has installed its handler.
            for _ in range(50):
                if root.handlers and isinstance(root.handlers[0], RichHandler):
                    break
                time.sleep(0.02)
            assert isinstance(root.handlers[0], RichHandler)
            logging.getLogger("mini.test").warning("mid-run-log")
            queue.put(ProgressMessage(run_id="r", job_id="j", step=1, total=1))
            queue.put(EndOfQueue())
        # After exit, original handlers are restored.
        assert root.handlers == saved_handlers
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)

    assert "mid-run-log" in buf.getvalue()

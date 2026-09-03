"""Tests for the RichProgressDisplay logging integration."""

from __future__ import annotations

import contextlib
import io
import logging
import time

import pytest

from rich.console import Console
from rich.logging import RichHandler

from mini._queues import EndOfQueue
from mini.local_queue import LocalQueue
from mini.progress import ProgressMessage
from mini.progress_display import RichProgressDisplay, _route_logging_to


@pytest.mark.parametrize("body_raises", [False, True], ids=["clean-exit", "body-raises"])
def test_route_logging_to_swaps_and_restores_root_handlers(body_raises: bool):
    """Root handlers are replaced inside the context and restored on exit — including when the body raises, since a display torn down by an error must not leave the root logger pointed at a dead console."""
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    saved = root.handlers[:]
    root.handlers = [sentinel]
    try:
        console = Console(file=io.StringIO(), force_terminal=False)
        with contextlib.suppress(RuntimeError), _route_logging_to(console):
            assert root.handlers != [sentinel]
            assert len(root.handlers) == 1
            assert isinstance(root.handlers[0], RichHandler)
            if body_raises:
                raise RuntimeError("boom")
        assert root.handlers == [sentinel]
    finally:
        root.handlers = saved


def test_rich_progress_display_routes_logging_while_running():
    """While the live display is running, logging goes through its console: the handler is installed by the display thread, records land in the display's buffer, and the caller's own handlers come back on exit."""
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

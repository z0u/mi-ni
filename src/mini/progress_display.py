"""
Display and aggregate progress messages using Rich.

This module provides a live progress display for apparatus by collecting
ProgressMessage objects via a queue and rendering them with Rich.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty
from typing import Self

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from mini._queues import EndOfQueue, QueueLike
from mini.local_queue import LocalQueue
from mini.progress import ProgressMessage


@contextmanager
def _route_logging_to(console: Console):
    """Route root-logger output through a Rich console for the duration.

    Rich's Live (used by Progress) repaints the bar after each console write,
    so log records emitted via this handler appear above a stable progress bar
    instead of clobbering it. Stdout/stderr are already redirected by Progress
    itself (``redirect_stdout``/``redirect_stderr`` default to True).

    Existing root handlers (e.g. those installed by ``mini.logging``) are
    restored on exit; per-logger levels are untouched.
    """
    root = logging.getLogger()
    handler = RichHandler(console=console, show_path=False, rich_tracebacks=True, markup=False)
    saved_handlers = root.handlers[:]
    root.handlers = [handler]
    try:
        yield
    finally:
        root.handlers = saved_handlers


def _is_in_notebook() -> bool:
    """Detect if we're running in a notebook-like environment (Jupyter, IPython, Marimo, etc.)."""
    try:
        from IPython.core.getipython import get_ipython  # type: ignore

        if get_ipython() is not None:
            return True
    except ImportError:
        pass

    try:
        import marimo as mo

        if mo.running_in_notebook():
            return True
    except ImportError:
        pass

    return False


@dataclass
class JobState:
    """State of a single job."""

    step: int = 0
    total: int = 0
    message: str = ""
    task_id: TaskID | None = None


class RichProgressDisplay:
    """
    Collect progress messages from a queue and display them using Rich.

    This runs in a background thread, periodically polling the queue for
    new progress messages and updating the Rich display.
    """

    queue: QueueLike[ProgressMessage]

    def __init__(self, total_jobs: int, queue: QueueLike[ProgressMessage] | None = None):
        self.total_jobs = total_jobs
        self.queue = queue or LocalQueue()
        self.jobs: dict[str, JobState] = {}
        self._any_message = threading.Event()
        if _is_in_notebook():
            self.console = Console(force_terminal=True)
        else:
            self.console = Console()
        self.progress: Progress | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    async def __aenter__(self) -> Self:
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the display, running the drain in a worker thread to avoid sync-in-async warnings."""
        await asyncio.to_thread(self.stop)

    def start(self) -> None:
        """Start the background display thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=False)
        self._thread.start()

    def stop(self, drain_timeout: float = 60.0) -> None:
        """
        Stop the background display thread.

        Wait up to *drain_timeout* seconds for all jobs to report completion
        before signalling the thread to exit.  This prevents the display from
        tearing down before in-flight progress messages arrive.
        """
        self.queue.put(EndOfQueue(), timeout=drain_timeout)
        self._stop_event.set()
        # Wake any threads waiting on _any_message (e.g. the startup watchdog)
        # so they don't block process exit via the ThreadPoolExecutor atexit handler.
        self._any_message.set()
        if self._thread:
            self._thread.join(timeout=drain_timeout)
        self.console.file.flush()

    def _run(self) -> None:
        """Main loop for the display thread."""
        with (
            Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress,
            _route_logging_to(progress.console),
        ):
            self.progress = progress
            self._completed = 0
            self._overall_task = progress.add_task(
                "[cyan]Overall progress[/]",
                total=self.total_jobs or None,
            )

            while True:
                try:
                    msg = self.queue.get(timeout=1.0)
                    self._any_message.set()
                    self._update_job(msg)
                except EndOfQueue:
                    break
                except Empty:
                    if self._stop_event.is_set():
                        break
                    continue

    def _update_job(self, msg: ProgressMessage) -> None:
        """Apply a single progress message to the display."""
        assert self.progress is not None
        job_id = msg.job_id

        if job_id not in self.jobs:
            task_id = self.progress.add_task(
                f"[cyan]Job {job_id}[/]",
                total=msg.total if msg.total > 0 else None,
            )
            self.jobs[job_id] = JobState(task_id=task_id, total=msg.total)

        state = self.jobs[job_id]
        state.step = msg.step
        state.message = msg.message
        if msg.total > 0 and state.total != msg.total:
            state.total = msg.total

        if state.task_id is not None and state.total > 0:
            desc = f"[cyan]Job {job_id}[/]"
            if state.message:
                desc += f" — {state.message}"
            self.progress.update(
                state.task_id,
                completed=state.step,
                total=state.total,
                description=desc,
            )

        if msg.step >= msg.total and msg.total > 0:
            self._completed += 1
            self.progress.update(self._overall_task, completed=self._completed)

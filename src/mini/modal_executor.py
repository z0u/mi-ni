"""
Modal executor for running experiment sweeps on Modal infrastructure.

Example::

    import modal
    from mini.modal_executor import ModalExecutor

    app = modal.App("my-experiment")
    executor = ModalExecutor(app).with_modal_kwargs(gpu="T4", timeout=3600)
    results = list(executor.map(train, configs))
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Iterator, TypeVar

import modal

from mini.executor import ProgressDisplay
from utils.requirements import freeze, project_packages

log = logging.getLogger(__name__)

R = TypeVar('R')

__all__ = ['ModalExecutor']


def make_app(name: str) -> modal.App:
    """Helper to create a Modal app with appropriate defaults for experiment runs."""
    # image = modal.Image.debian_slim().pip_install(*freeze(all=True)).add_local_python_source(*project_packages())
    image = (
        modal.Image.debian_slim()
        .pip_install(*freeze(all=True, local=False))
        .add_local_python_source(*project_packages())
    )  # fmt: skip
    return modal.App(name, image=image)


class ModalExecutor:
    """
    Run functions on Modal.

    Fine-grained progress (epoch-level etc.) is visible through Modal's
    built-in log streaming --- mapped functions can simply ``print()``.
    Job-level completion is tracked locally via a progress display.

    Usage::

        app = make_app("my-experiment")
        executor = ModalExecutor(app).with_modal_kwargs(gpu="T4", timeout=3600)
        results = list(executor.map(train, configs))
    """

    def __init__(self, app: modal.App, modal_fn_kwargs: dict[str, Any] | None = None):
        self.app = app
        self.modal_fn_kwargs: dict[str, Any] = modal_fn_kwargs or {}

    def with_modal_kwargs(self, **kwargs: Any) -> ModalExecutor:
        """Return a new executor with additional Modal function kwargs merged in."""
        return ModalExecutor(self.app, {**self.modal_fn_kwargs, **kwargs})

    def map(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> Iterator[R]:
        iterables_lists: list[list] = [list(it) for it in iterables]
        n = len(iterables_lists[0]) if iterables_lists else 0
        if n == 0:
            return

        log.info('[ModalExecutor] Running %d jobs on Modal', n)

        # Wrap fn as a Modal function.  The decorator must be applied *before*
        # app.run() starts the app.
        modal_fn = self.app.function(**self.modal_fn_kwargs)(fn)

        display = ProgressDisplay(n)
        # We don't have per-step progress from Modal, but we can track job-level
        # completion as results come back.
        with modal.enable_output(), self.app.run():
            try:
                for i, result in enumerate(modal_fn.map(*iterables_lists, kwargs=kwargs or {})):
                    display.job_completed(i)
                    yield result
            finally:
                display.finish()

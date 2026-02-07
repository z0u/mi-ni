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

from functools import wraps
import logging
from typing import Any, Callable, Iterable, Iterator, TypeVar, override

import modal

from mini.executor import Executor, ProgressDisplay
from utils.requirements import freeze, project_packages

log = logging.getLogger(__name__)

R = TypeVar('R')

__all__ = ['ModalExecutor']


def make_app(name: str) -> modal.App:
    """Helper to create a Modal app."""
    return modal.App(name)


def make_image() -> modal.Image:
    """Helper to create a Modal image with experiment dependencies."""
    return (
        modal.Image.debian_slim()
        .pip_install(*freeze(all=True, local=False))
        .add_local_python_source(*project_packages())
    )  # fmt: skip


class ModalExecutor(Executor):
    """
    Run functions on Modal.

    Fine-grained progress (epoch-level etc.) is visible through Modal's
    built-in log streaming --- mapped functions can simply ``print()``.
    Job-level completion is tracked locally via a progress display.

    Usage::

        executor = ModalExecutor("my-experiment").w(gpu="T4", timeout=3600)
        results = list(executor.map(train, configs))
    """

    app: modal.App

    def __init__(self, app: modal.App | str):
        self.app = modal.App(app) if isinstance(app, str) else app
        self.modal_fn_kwargs: dict[str, Any] = {
            'image': make_image(),
        }
        self._before_hooks: list[Callable[[], None]] = []

    def clone(self) -> ModalExecutor:
        new_executor = ModalExecutor(self.app)
        new_executor.modal_fn_kwargs = self.modal_fn_kwargs.copy()
        new_executor._before_hooks = self._before_hooks[:]
        return new_executor

    def w(self, **kwargs: Any) -> ModalExecutor:
        """
        Return a new executor with additional Modal function kwargs merged in.

        These kwargs are passed to the ``@app.function()`` decorator when
        mapping, and can be used to specify things like GPU requirements or
        timeouts.
        """
        new_executor = self.clone()
        new_executor.modal_fn_kwargs = {**self.modal_fn_kwargs, **kwargs}
        return new_executor

    @override
    def before_each(self, hook: Callable[[], None]) -> ModalExecutor:
        new_executor = self.clone()
        new_executor._before_hooks = self._before_hooks + [hook]
        return new_executor

    @override
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

        hooks = self._before_hooks
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            for hook in reversed(hooks):
                hook()
            return fn(*args, **kwargs)

        # Wrap fn as a Modal function.  The decorator must be applied *before*
        # app.run() starts the app.
        modal_fn = self.app.function(**self.modal_fn_kwargs)(wrapped_fn)

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

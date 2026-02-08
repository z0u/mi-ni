"""
Executor protocol and progress reporting infrastructure.

Provides shared components for executor implementations:
- ``Executor`` protocol defining the map interface
- ``ProgressDisplay`` for tracking concurrent job progress
- ``get_progress()`` for mapped functions to report progress

Example::

    from mini.executor import get_progress

    def train(params):
        progress = get_progress()
        if progress:
            progress.set_total(100)
        for epoch in range(100):
            ...
            if progress:
                progress.update(1, message=f"loss={loss:.4f}")
        return result
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Iterable, Iterator, ParamSpec, TypeVar

P = ParamSpec('P')
R = TypeVar('R')


# ---------------------------------------------------------------------------
# Executor protocol
# ---------------------------------------------------------------------------


class Executor(ABC):
    """Protocol for running a function over a sweep of inputs."""

    def run(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Run a single function and return its result."""

        @wraps(fn)
        def wrapper(_) -> R:
            return fn(*args, **kwargs)

        return next(self.map(wrapper, [None]))

    @abstractmethod
    def map(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> Iterator[R]:
        """
        Map *fn* over one or more iterables.

        Like ``concurrent.futures.Executor.map`` and Modal's ``Function.map``:
        the iterables are zipped together and each tuple is unpacked as
        positional arguments.  *kwargs* (if given) are forwarded to every
        call.

        ::

            executor.map(fn, [1, 2, 3])                    # fn(1), fn(2), fn(3)
            executor.map(fn, [1, 2], ['a', 'b'])            # fn(1, 'a'), fn(2, 'b')
            executor.map(fn, [1, 2], kwargs={'k': 'v'})     # fn(1, k='v'), fn(2, k='v')
        """
        ...

    @abstractmethod
    def before_each(self, hook: Callable[[], None]) -> Executor:
        """
        Return a new executor that runs *hook* before each job.

        This is useful for things like configuring logging or setting random
        seeds on a per-job basis.
        """
        ...

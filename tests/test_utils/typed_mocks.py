from typing import Awaitable, Callable, ParamSpec, TypeVar
from unittest.mock import AsyncMock, Mock


P = ParamSpec('P')
R = TypeVar('R')


def typed_mock(wraps: Callable[P, R] | None = None) -> Callable[P, R] | Mock:
    """Create a mock function with a particular signature."""
    return Mock(wraps=wraps)


def typed_async_mock(wraps: Callable[P, Awaitable[R]] | None = None) -> Callable[P, Awaitable[R]] | AsyncMock:
    """Create an async mock function with a particular signature."""
    return AsyncMock(wraps=wraps)

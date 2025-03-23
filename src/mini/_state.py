from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import Literal, TypeAlias

from mini.urns import parse_urn, matches_urn, to_urn

log = logging.getLogger(__name__)


FunctionState: TypeAlias = Literal['guard', 'start', 'error', 'end']
STATES = ('guard', 'start', 'error', 'end')


@dataclass
class CallState:
    run_id: str
    fn_name: str
    fn_id: str
    call_id: str
    state: FunctionState
    msg: str = ''

    def __str__(self) -> str:
        return self.to_urn()

    def to_urn(self) -> str:
        """Convert to a URN."""
        return to_urn(
            'mini', 'run', self.run_id, 'fn', self.fn_name, self.fn_id, 'call', self.call_id, self.state, self.msg
        )

    @classmethod
    def matches(cls, message: str) -> bool:
        return matches_urn(message, 'mini:run:*:fn:*:*:call:*:*')

    @classmethod
    def from_urn(cls, message: str) -> CallState:
        """Convert from a URN."""
        parts = parse_urn(message)
        match parts:
            case ('mini', 'run', run_id, 'fn', fn_name, fn_id, 'call', call_id, state, msg) if state in STATES:
                return cls(run_id=run_id, fn_name=fn_name, fn_id=fn_id, call_id=call_id, state=state, msg=msg)
            case _:
                raise ValueError(f'Invalid call state format: {message}')


class CallTracker:
    """Tracks the state of remotely executing functions."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.calls: dict[str, FunctionState] = {}
        self.state_counts: dict[FunctionState, int] = defaultdict(int)
        self.state_history: list[CallState] = []

    def handle(self, state: CallState) -> None:
        """Update a function's state."""
        self.state_history.append(state)

        allowed_from = VALID_TRANSITIONS.get(state.state)
        if allowed_from is None:
            raise CallStateError(f'Invalid state: {state.state}')

        prev_state = self.calls.get(state.call_id)
        log.debug(f'Function {state.fn_name} ({state.fn_id}:{state.call_id}) state: {prev_state} -> {state.state}')
        if prev_state not in allowed_from:
            raise CallStateError(f'Invalid state transition: {prev_state} -> {state.state} ({state})')

        if prev_state is not None:
            self.state_counts[prev_state] -= 1
        self.state_counts[state.state] += 1
        self.calls[state.call_id] = state.state

    def any_active(self) -> bool:
        """Check if any functions are not finished."""
        return any(state != 'end' for state in self.calls.values())

    def any_running(self) -> bool:
        """Check if any functions are in the 'start' state."""
        return self.state_counts['start'] > 0


class CallStateError(Exception):
    pass


VALID_TRANSITIONS: dict[FunctionState, tuple[FunctionState | None, ...]] = {
    'guard': (None,),
    'start': ('guard',),
    'error': ('guard', 'start'),
    'end': ('start', 'error'),
}
"""Transitions (to: from)"""

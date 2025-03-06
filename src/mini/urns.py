from itertools import zip_longest
import re
import urllib.parse
from uuid import uuid4 as uuid


mini_urn_pattern = re.compile(r'^mini(?::[a-zA-Z0-9%._-]*)+$')


def is_mini_urn(urn: str) -> bool:
    """Check if a URN is a mini URN."""
    return bool(mini_urn_pattern.match(urn))


def matches_urn(urn: str, pattern: str) -> bool:
    """
    Check that a URN matches a pattern.

    The pattern can contain wildcards ('*'); these match anything except a colon.
    """
    parsed = parse_urn(urn)
    partspec = pattern.split(':')
    for part, spec in zip_longest(parsed, partspec):
        if spec == '*':
            # Allow wildcard
            continue
        if spec is None:
            # Pattern is shorter than URN
            break
        if part != urllib.parse.unquote(spec):
            return False
    return True


def to_urn(*parts: str) -> str:
    """Convert a sequence of parts to a colon-separated URN."""
    return ':'.join(urllib.parse.quote(part, safe='') for part in parts)


def parse_urn(urn: str) -> tuple[str, ...]:
    """Convert a URN to a tuple of parts."""
    parts = urn.strip().split(':')
    return tuple(urllib.parse.unquote(part) for part in parts)


def short_id() -> str:
    """Generate a short UUID for use in URNs."""
    return str(uuid())[:8]

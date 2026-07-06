import re


units = {"ms": 0.001, "s": 1, "m": 60, "min": 60, "h": 3600, "d": 86400}


def duration(d: str) -> float:
    """
    Convert a string duration to seconds.

    Supported units: `ms`, `s`, `m`/`min`, `h`, `d`.

    Example:
        ```python
        >>> duration('1h')
        3600.0
        >>> duration('2 min')
        120.0
        >>> duration('30m')
        1800.0
        ```
    """
    # Order matters: `ms`/`min` must be tried before the single-char `m`/`s`.
    match = re.match(r"([+-]?[0-9.]+(?:[eE][+-]?[0-9]+)?) ?(ms|min|m|s|h|d)", d, re.IGNORECASE)
    if match:
        value, unit = match.groups()
        return float(value) * units[unit.lower()]
    raise ValueError(f"Invalid duration format: {d}")

import re


units = {'ms': 0.001, 's': 1, 'min': 60, 'h': 3600, 'd': 86400}


def duration(d: str) -> float:
    """
    Convert a string duration to seconds.

    Supported units: `ms`, `s`, `min`, `h`, `d`.

    Example:
        ```python
        >>> duration('1h')
        3600.0
        >>> duration('2 min')
        120.0
        ```
    """
    match = re.match(r'([+-]?[0-9.]+(?:[eE][+-]?[0-9]+)?) ?(ms|s|min|h|d)', d, re.IGNORECASE)
    if match:
        value, unit = match.groups()
        return float(value) * units[unit]
    raise ValueError(f'Invalid duration format: {d}')

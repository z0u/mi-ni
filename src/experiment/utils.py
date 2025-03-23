import math


def align(x: int, y: int) -> int:
    """Align x to the nearest multiple of y."""
    return int(math.ceil(x / y) * y)

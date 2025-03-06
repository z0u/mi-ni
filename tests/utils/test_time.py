import pytest

from utils.time import duration


@pytest.mark.parametrize(
    'input, expected',
    [
        ('1h', 3600.0),
        ('2 min', 120.0),
        ('3s', 3.0),
        ('4.5d', 388800.0),
        ('5ms', 0.005),
        ('6e-3s', 0.006),
        ('7e+2min', 42000.0),
    ],
)
def test_duration(input, expected):
    assert duration(input) == expected

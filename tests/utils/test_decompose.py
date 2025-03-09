from utils.decompose import select, selecta, sliceable


def test_select():
    data = {"a": 1, "b": 2, "c": 3, "d": 4}

    # Single key access
    assert (2,) == select(data, "b")

    # Multiple keys with spaces
    assert (1, 3) == select(data, "a c")

    # Multiple keys as list
    assert (1, 3) == select(data, ["a", "c"])

    # Handles various string formats
    assert (1, 3) == select(data, "a,c")
    assert (1, 3) == select(data, "a, c")

    # Missing keys return None
    assert (1, None, 3) == select(data, "a x c")

    # Works with custom objects
    class Config:
        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg = Config()
    cfg.host = "localhost"
    cfg.port = 5432
    assert ("localhost", 5432) == select(cfg, "host,port")


def test_selecta():
    from collections import namedtuple

    Foo = namedtuple("Foo", "a,b,c,d")
    data = Foo(1, 2, 3, 4)

    # Single key access
    assert (2,) == selecta(data, "b")

    # Multiple keys with spaces
    assert (1, 3) == selecta(data, "a c")

    # Multiple keys as list
    assert (1, 3) == selecta(data, ["a", "c"])

    # Handles various string formats
    assert (1, 3) == selecta(data, "a,c")
    assert (1, 3) == selecta(data, "a, c")

    # Missing keys return None
    assert (1, None, 3) == selecta(data, "a x c")


def test_sliceable():
    from collections import namedtuple

    Foo = sliceable(namedtuple("Foo", "a,b,c,d"))

    foo = Foo(1, 2, 3, 4)
    assert (2,) == foo["b"]
    assert (1, 3) == foo["a c"]
    assert (1, 3) == foo[["a", "c"]]
    assert (1, 3) == foo[[0, 2]]
    assert (2, 3, 4) == foo["b:d"]
    assert foo == (1, 2, 3, 4)

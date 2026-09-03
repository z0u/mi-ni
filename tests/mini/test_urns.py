from mini.urns import matches_urn


def test_matches_exact():
    """Test exact matching without wildcards."""
    assert matches_urn("mini:test:123", "mini:test:123")
    assert not matches_urn("mini:test:123", "mini:test:456")
    assert not matches_urn("mini:foo:bar", "mini:foo:baz")


def test_matches_with_wildcards():
    """Test matching with wildcards."""
    assert matches_urn("mini:test:123", "mini:*:123")
    assert matches_urn("mini:test:123", "mini:test:*")
    assert matches_urn("mini:test:123", "*:test:123")
    assert matches_urn("mini:test:123", "*:*:*")
    assert not matches_urn("mini:test:123", "mini:*:456")

    # Several wildcards in one pattern.
    assert matches_urn("mini:foo:bar:baz", "mini:*:bar:*")
    assert matches_urn("mini:foo:bar:baz:qux", "mini:*:*:baz:*")
    assert not matches_urn("mini:foo:bar:baz", "mini:*:qux:*")


def test_matches_different_lengths():
    """Test patterns with different lengths than the URN."""
    assert not matches_urn("mini:test:123", "mini:test:123:extra")
    assert matches_urn("mini:test:123:extra", "mini:test:123")
    assert matches_urn("mini:test:123:extra", "mini:test:*")


def test_matches_with_url_encoding():
    """Test with URL-encoded components."""
    # The encoded URN for "mini:test:hello world"
    encoded_urn = "mini:test:hello%20world"
    assert matches_urn(encoded_urn, "mini:test:hello%20world")
    assert matches_urn(encoded_urn, "mini:*:*")
    assert not matches_urn(encoded_urn, "mini:test:hello")

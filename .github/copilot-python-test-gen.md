This project uses pytest.

Use pytest idioms: fixtures, parametrize, assert.
Prefer functional tests (not class-based).
Prefer pytest-native tools
Prefer brevity where possible
Use structural assertions

---

## Examples illustrated as diffs

```diff
+ from pytest import approx

- np.isclose(x, y)  # ❌
+ pytest.approx(x) == y  # ✅
```

```diff
+ from unittest.mock import ANY

- assert 'x' in props and approx(props['x']) == 1.0  # ❌
- assert 'z' in props and approx(props['z']) == 0.8  # ❌
+ assert approx(props) == {'x': 1.0, 'y': ANY, 'z': 0.8}  # ✅
```

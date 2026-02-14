---
name: testing
description: Guidelines for writing tests in this project. Patterns, what to test, how to write assertions.
---

# Writing tests

This project uses pytest.

Use pytest idioms: fixtures, parametrize, assert.
Prefer functional tests (not class-based).
Prefer brevity where possible.

Prefer pytest-native tools.

```diff
+ from pytest import approx

- np.isclose(x, y)  # ❌
+ pytest.approx(x) == y  # ✅
```

Use structural assertions.

```diff
+ from unittest.mock import ANY

- assert 'x' in props and approx(props['x']) == 1.0  # ❌
- assert 'z' in props and approx(props['z']) == 0.8  # ❌
+ assert approx(props) == {'x': 1.0, 'y': ANY, 'z': 0.8}  # ✅
```

Use reserved domains to avoid accidentally fetching from real domains: `.example`, `.test`, `.invalid`.

```diff
- response = requests.get('test.com')  # ❌ this is a real domain!
+ response = requests.get('service.test')  # ✅ guaranteed not to resolve
```

Use explicit, literal pre-conditions where possible:

```diff
- input = np.arange(5) * 2  # ❌ have to mentally evaluate this
+ input = [0, 2, 4, 6, 8]  # ✅ immediately clear what the input is
```

Use explicit, literal expected values where possible:

```diff
  output = add(a, b)
- assert output == a + b  # ❌ tautological assertion; doesn't verify anything
+ assert output == 5  # ✅ verifies that the function produces the expected result
```

Use `pytest.mark.parametrize` to test multiple cases without repetition.

## What to test?

**We only write valuable tests.** We test for behavioral verification under uncertainty:

- Exercise meaningful state transitions and invariants: Tests that verify your system maintains its promises; Boundary condition handling; State consistency across operations (e.g., after a series of mutations, derived state still makes sense)
- Capture domain logic and business rules: Scenarios that encode actual user workflows or data processing pipelines; Edge cases that reflect real-world complexity your system needs to handle
- Reveal integration assumptions: How your code behaves when dependencies return unexpected (but valid) responses; Error propagation and recovery behavior; Resource cleanup and lifecycle management
- Executable documentation: Tests that demonstrate intended usage patterns, with clear naming that explains the "given/when/then" story

Valuable tests fail _for interesting reasons_ — they break when you've actually broken something that matters to users. We rely on linters and type-checkers for everything else.

## Wrapping up

When you have finished, review your work with a critical eye. Ask yourself: Does this test actually verify something meaningful? Is it clear what the test is doing and why? Could it be simplified without losing value?

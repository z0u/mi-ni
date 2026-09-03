---
name: testing
description: Guidelines for running and writing tests in this project. Patterns, what to test, how to write assertions.
---

# Writing tests

This project uses pytest.

- Use pytest idioms: fixtures, parametrize, assert, approx, ANY.
- Prefer brevity.

## Fast, serial, hermetic

The whole suite runs in about 30s serially (`uv run pytest -n0`) and half that under the default `pytest-xdist` (`-n auto`). Keep it that way: mock or fake anything that sleeps, polls, or talks to the network (see `test_watchdog.py`'s fake clock and `test_orchestration.py`'s `drive_and_watch(poll=0.005)`), use toy models with the smallest config that still clears the assertion with margin, and isolate all state via `tmp_path`/`monkeypatch` — never write to a shared path or the cwd. Don't rely on test execution order. A per-test `timeout = 60` fails a wedged test rather than hanging the run. JAX's persistent compile cache is on (`tests/conftest.py`), so jitted steps compile once per checkout.

Tests that need a real service carry a marker and are deselected by default: `uv run pytest -m hf` runs the Hugging Face bucket tests (they probe for write access and skip on a read-only token). Run them when touching `src/mini/hf_store.py`.

Tests for `scripts/*.py` load the script with `load_script("name")` from `tests/conftest.py`.

Prefer specialized testing utilities, and specify tolerances.
```diff
- assert np.allclose(x, y)  # ❌
- assert torch.allclose(a, b)  # ❌
+ np.testing.assert_allclose(x, y, rtol=1e-7, atol=0)  # ✅
+ torch.testing.assert_close(a, b, rtol=1e-7, atol=0)  # ✅
```
Reason: Specialized testing utilities give better error messages when assertions fail, and they often have features that generic assertions lack. Tolerances are highly context-dependent, so choose them based on principle.

Use structural assertions.
```diff
+ from unittest.mock import ANY

- assert "x" in props and approx(props["x"]) == 1.0  # ❌
- assert "z" in props and approx(props["z"]) == 0.8  # ❌
+ assert approx(props) == {"x": 1.0, "y": ANY, "z": 0.8}  # ✅
```
Reason: Structural assertions that fail will show you the entire structure and all the differences, not just the first failed assertion. This makes it much easier to understand what went wrong.

Use reserved domains to avoid accidentally fetching from real domains: `.example`, `.test`, `.invalid`.
```diff
- response = requests.get("test.com")  # ❌ this is a real domain!
+ response = requests.get("service.test")  # ✅ guaranteed not to resolve
```
Reason: Using real resources in tests can lead to flaky tests and unintended side effects.

Use explicit, literal pre-conditions:
```diff
- input = np.arange(5) * 2  # ❌ have to mentally evaluate this
+ input = [0, 2, 4, 6, 8]  # ✅ immediately clear what the input is
```
Reason: Bugs can hide in complex test setup code.

Use explicit, closed-form analytical expected values:
```diff
  output = add(a, b)
- assert output == a + b  # ❌ tautological assertion; doesn't verify anything
+ assert output == 5  # ✅ verifies that the function produces the expected result
```
Reason: A tautological assertion could easily share the same bug as the code under test, and thus fail to catch it. Analytical expected values can be verified by hand.

Use `pytest.mark.parametrize` to test multiple cases without repetition.

## What to test?

We only write valuable tests. We test for behavioral verification under uncertainty:

- Exercise meaningful state transitions and invariants: Tests that verify your system maintains its promises; Boundary condition handling; State consistency across operations (e.g., after a series of mutations, derived state still makes sense)
- Capture domain logic and business rules: Scenarios that encode actual user workflows or data processing pipelines; Edge cases that reflect real-world complexity your system needs to handle
- Reveal integration assumptions: How your code behaves when dependencies return unexpected (but valid) responses; Error propagation and recovery behavior; Resource cleanup and lifecycle management
- Executable documentation: Tests that demonstrate intended usage patterns, with clear naming that explains the "given/when/then" story

Valuable tests fail for interesting reasons: they break when you've broken something that matters to users. We rely on linters and type-checkers for everything else.

## Wrapping up

When you have finished, review your work: does this test verify something meaningful? Is it clear what the test is doing and why? Could it be simplified without losing value?

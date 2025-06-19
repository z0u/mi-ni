### Style
Use JavaScript-style method chaining (newline before the dot, use outer parentheses as necessary).
Use cutting-edge syntax.
Prefer brevity.
Use single quotes for strings, except for multiline strings.

### Docstrings
Use the imperative for the first line of function docstrings.

```diff
- """Adds two numbers together"""
+ """Add two numbers together."""
```

### Typing
Use type hints.
Use `T | None` instead of `Optional[T]`.

```diff
- foo: Optional[int] = None
+ foo: int | None = None
```

### Modal
Use Modal-compatible patterns for distributed processing.
Returning a model from a remote training function may be infeasible if the model is large.
Most objects including custom functions and classes can be pickled and executed remotely.
Closures work even for remote functions, but don't _assume_ global scope.

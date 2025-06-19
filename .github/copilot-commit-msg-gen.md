Be as concise as possible.
Single-line commit messages are acceptable.
Don't just list out the changes; instead, try to capture the _intent_ of the commit.

# Examples

```diff
- updated README
+ doc: described setup process
```

```diff
- using update_display
+ nb: redrawing the plot with the same display handle
```

```diff
- added SmoothProp class
+ promoted transition logic to its own module
+
+ Experiments in the notebook went well, so copying the SmoothProp to its own
+ module so that it can be reused.
```

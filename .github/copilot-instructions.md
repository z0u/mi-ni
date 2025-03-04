We are writing code to run AI experiments.

Design the code well:

- Functions should lend themselves to concurrent and distributed processing.
- For example, returning a model from a training function may be infeasible if the model is large. Consider using remote storage.
- Most objects including custom functions and classes can be pickled and executed remotely.
- Jupyter Notebook cells should to be written accordingly: don't assume global scope.

However this is not production code, so prioritize quick iteration over robustness.

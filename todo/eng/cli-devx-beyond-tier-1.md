---
status: open
tags: [cli]
bundle: cli-devx
---
# CLI DevX beyond tier 1

Tracked as [#57](https://github.com/z0u/mi-ni/issues/57): passing a name to `retry`/`run` died with a raw traceback, because tick verbs take a file and read verbs a name.

Tier 1 shipped (`_load_experiment_or_hint` in `src/mini/__main__.py` gives a friendly error, and the `path` positional documents file-vs-NAME). Anything beyond that (e.g. auto-resolving a name to its experiment file) is still open.

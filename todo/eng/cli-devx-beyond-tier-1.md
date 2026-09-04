---
status: open
tags: [cli]
bundle: cli-devx
---
# CLI DevX beyond tier 1

Was tracked as [#57](https://github.com/z0u/mi-ni/issues/57), closed by [#58](https://github.com/z0u/mi-ni/pull/58) with tier 1: passing a name to `retry`/`run` died with a raw traceback, because tick verbs take a file and read verbs a name.

Tier 1 shipped (`_load_experiment_or_hint` in `src/mini/__main__.py` gives a friendly error, and the `path` positional documents file-vs-NAME). Anything beyond that (e.g. auto-resolving a name to its experiment file) is still open.

The rest of the same cold-exploration session is its own items in the `cli-devx` bundle: [`mini logs` and the `fc-…` ids](./mini-logs-and-fc-ids.md), [`mini ls` on Modal](./mini-ls-remote-experiments.md), [`mini rm`](./mini-rm-experiment-memo-state.md), and the [`watch` flicker](./mini-watch-ui-flicker.md).

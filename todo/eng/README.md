# Engineering todo

Deferred infrastructure work: tooling, storage, publishing, CLI, and the `mini` library. Experiment questions live in [`science/`](../science/), text and visual improvements in [`style/`](../style/). The schema and the `./go todo` query are described once, in [`todo/README.md`](../README.md).

Durable design rationale and recorded decisions live in [`eng/`](/eng/README.md) — an item that grows into a design doc belongs there, with a line here pointing at it.

Larger work is tracked as a [GitHub issue](https://github.com/z0u/mi-ni/issues); an item that stands in for one links it by full URL rather than a bare `#38`, which only auto-links inside issues and pull requests. Each open issue carries a grounding comment with current file:line refs, so it should be readable cold without re-deriving code state.

Projects started from this template inherit the `mini` library, so a library bug found downstream is usually fixed there first and backported: an item here that records one says so, and points at the tree it came from.

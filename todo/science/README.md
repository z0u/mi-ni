# Science todo

Experiment questions and findings. Add to it when you notice something interesting in a run but want to defer the investigation. Infrastructure and tooling work lives in [`eng/`](../eng/); the schema and the `./go todo` query are described once, in [`todo/README.md`](../README.md).

Two kinds of item share this directory. An **open question** is work: something to run or decide. A **finding** (`status: finding`) is established knowledge with no completion state — a result worth carrying into the next experiment. `./go todo science` shows the first; `./go todo science --status finding` shows the second.

Items are tagged by the experiment they came from, the one they bear on, and the concepts they touch. A project that adopts this template should write its vocabulary down here, one heading per tag, so a new item can pick from a list rather than coin a near-duplicate. Deliverable and milestone tags are capitalised (`D1.2`, `M2`); the rest are lower-case.

## Tags

(none yet — the demo reports under `docs/` did not need any; add the project's own here)

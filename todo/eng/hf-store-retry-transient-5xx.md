---
status: open
tags: [storage, mini, reliability]
opened: 2026-09-16
---
# Retry transient 5xx responses on bucket batch writes

`HfStore._write_blob` makes one `batch_bucket_files` call per put with no retry, so a single transient server error fails the whole task. In the ex-2.2.9 run on 2026-09-16, the Hugging Face bucket's `/batch` endpoint returned `500 Internal Server Error` twice in twenty minutes: once inside a `score_one` (four minutes of compute lost) and once inside `publish_results`, the last step of the DAG. Both cleared on `bin/mini retry`, at the cost of a driver relaunch and a human noticing.

A bounded retry with exponential backoff (three or four attempts over a minute, on 5xx and connection errors only, never on 4xx) around the batch add in `_write_blob`, and probably around `_pull_blobs` and `_paths_info`, would absorb this. Keep it inside `hf_store.py` so the `LocalStore` path and the memo semantics stay untouched: the retry wraps one idempotent HTTP call (a CAS write keyed by content hash), so a duplicate attempt cannot corrupt anything.

Worth checking first whether `huggingface_hub` already exposes a retry knob for bucket calls; if it does, turning that on is the smaller change.

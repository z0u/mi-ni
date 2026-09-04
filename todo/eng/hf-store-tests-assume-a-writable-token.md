---
status: done
tags: [testing, storage]
opened: 2026-08-25
closed: 2026-08-27
---
# The HF store tests gate on a token existing, not on it being able to write

`tests/mini/test_hf_store.py` skips itself unless `MINI_STORE_BUCKET` and `HF_TOKEN` are both set. That gate reads as "can I reach a real bucket", and it was right while the only tokens around were writable ones. A read-only token satisfies it and then fails seven tests on a 403 from the first write, which is what a Claude Code web session sees: those sessions are handed a read-only `HF_TOKEN` on purpose, so `./go check` there ends `7 failed, 827 passed` every single time, on any branch, for a reason that has nothing to do with the branch.

Seen on 2026-08-25 while verifying an unrelated CI change, and confirmed against a clean checkout of `main` so it isn't the branch's doing. The cost isn't the seven minutes; it's that a session which has learnt to read past a red test suite has lost the signal for a real one, and the next genuine failure lands in the same paragraph.

The gate wants to ask the question it means. A one-off write probe in a session fixture — write a tiny blob under the `_test/` prefix the suite already cleans up, skip the module on a 403 — answers it directly and costs one round trip, only when both variables are already set. Cheaper and cruder: an explicit `MINI_STORE_TEST_WRITE=1` opt-in, which never lies but does have to be remembered by whoever has the writable token. Either way the skip reason should say "the token can't write to the bucket" rather than "set the variables", since the variables *are* set.

Not fixed on the spot because the positive path can't be exercised from a session with a read-only token: a probe that wrongly skips everywhere would be worse than the present noise, and proving it doesn't needs a writable token to hand.

## Resolution (2026-08-27)

Both halves. The gate now resolves the bucket, publish repo, and token the way `mini.store` does (env, else `[tool.mini]`, else the `hf auth login` cache), and a module-scoped fixture writes and deletes one tiny ref before any case runs: a 401/403 there skips the module with "the HF token can't write to <bucket>". The positive path was exercised from a session holding a writable cached token (7 passed), so the probe doesn't wrongly skip. Separately, the module now carries the `hf` marker and is deselected by default — each bucket commit is a 2-3s round trip, ~90s for the file, which has no place in a suite meant to run fast serially. `uv run pytest -m hf` opts in; do that when touching `src/mini/hf_store.py`.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/hf-store-tests-assume-a-writable-token.md`](https://github.com/z0u/sca2/blob/main/todo/eng/hf-store-tests-assume-a-writable-token.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

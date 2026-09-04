---
status: done
tags: [ci, publishing]
opened: 2026-08-31
closed: 2026-09-02
---
# Seven closed PRs still have their previews on the site

Found while checking that the first real `gh-pages` prune had left the served tree intact — it had, and this was sitting in it.

`pr-preview.yml` tears a preview down when its PR closes, and mostly does: of 112 `pr-preview/pr-N/` directories on the branch, 103 hold nothing but a lone `.nojekyll` stub, which is what a completed teardown leaves. Seven hold a whole site build for a PR that closed weeks ago — **123, 112, 65, 59, 51, 42, 37** — and one more pair (127, 131) is legitimately live. Measured 2026-08-31: the preview tree is 30.8 MiB of the site's 36.7 MiB, and the seven stale ones are about 25 MiB of that. So roughly two thirds of what the site serves is previews of work that has already landed.

The bytes are the smaller half. Each is a reachable URL serving a report at a revision nobody promoted, and this is a project where the figures are the argument — a link to `pr-preview/pr-65/` shows science that was superseded, with nothing on the page saying so.

Why the teardown didn't run is the part worth establishing before deleting anything, because the answer decides whether this recurs. Candidates, roughly in order of how easy each is to check: the PR closed before `closed` was in the workflow's trigger `types`; the teardown run was cancelled by the `preview-<ref>` concurrency group (a group holds one pending run and a newer arrival displaces it); or the run failed. The teardown commits that would settle it are no longer on the branch — the prune keeps 3 commits — but the workflow run history for `pr-preview.yml` still has them, and the pattern of *which* PRs are affected should be legible from their close dates.

Not deleting them without a decision: it is an outward-facing change to a published site, the cause is unestablished, and a wrong sweep removes a preview someone is reading. The fix is probably a one-off cleanup plus whichever guard the cause implies — and if the cause turns out to be concurrency cancellation, that is also evidence for the note in [`eng/publishing.md`](/eng/publishing.md) about not putting the previews and the production deploy in a shared group, where the same cancellation would reach teardowns far more often.

## Notes

**2026-09-02, closing** — Cause established from the run history (in [sca2#136](https://github.com/z0u/sca2/pull/136), closed unmerged): two of the seven were a teardown overwritten by a `synchronize` build already in flight, the two sitting in different concurrency groups since a `closed` event's `github.ref` isn't the one its builds use; three were a `closed` event that scheduled no run at all. Rather than fix the first and sweep for the second, the site is now rebuilt whole on every event by `site.yml` and deployed as one commit (`scripts/deploy_site.py`), so the first run removes all seven and there is no teardown left to miss. The reasoning is in [`eng/publishing.md`](/eng/publishing.md).

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/stale-pr-previews-on-the-site.md`](https://github.com/z0u/sca2/blob/main/todo/eng/stale-pr-previews-on-the-site.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on; the PR numbers are that project's.

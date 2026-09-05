---
status: open
tags: [tooling, storage]
opened: 2026-09-05
bundle: backup-template
---
# A queued backup run can't record itself: the push is rejected on a stale parent

Dispatching several backup runs at once — the way to work through a long first `pub/` replay — leaves every run after the first unable to push its `state/` commit: `! [rejected] HEAD -> main (fetch first)`. The backup itself is unharmed, but the run's record is lost, and with it the repository activity that keeps a public repo's schedule from being disabled after 60 days.

`workflow_dispatch` pins a run to the sha its branch had when it was dispatched, and `actions/checkout` takes `github.sha` rather than the current tip. The `concurrency` group serialises when the runs *execute*, and not what they checked out, so every queued run is committing on the same stale parent; the first to finish pushes and the rest are refused. A lone nightly run never meets this.

The fix is in `templates/backup/.github/workflows/backup.yml`, in the `Record the run` step: rebase onto whatever landed meanwhile and retry. `-X theirs` during a rebase favours the commit being replayed, so the newer run's record wins, which is the precedence we want.

```yaml
          git add -A state/
          git commit -q -m "backup $(date -u +%F)" || exit 0
          attempt=0
          until git push origin "HEAD:$GITHUB_REF_NAME"; do
            attempt=$((attempt + 1))
            if [ "$attempt" -gt 3 ]; then exit 1; fi
            git fetch origin "$GITHUB_REF_NAME"
            git rebase -X theirs FETCH_HEAD
          done
```

Write the retry limit as `if [ … ]; then exit 1; fi` rather than `[ … ] && exit 1`: Actions runs the step under `bash -e`, where the `&&` form returns 1 when the test is false and aborts the step. Verified against two clones sharing a parent — clean push, rejected-then-rebased push, and nothing-to-commit all behave.

Re-running after a rejection is safe meanwhile, and costs no repeated work: every leg reads its resume point from its target rather than from `state/`. `pub` reads `pub/SOURCE_COMMIT`, written into the same commit as each replayed revision; `code` reads the mirror's `mirror` branch; `store` diffs the two bucket listings. Only `store-missing.json` lives solely in `state/`, and losing it delays expiry rather than hurrying it.

## Notes

**2026-09-05, setup** — Met while installing the template into sca2's backup repo ([z0u/sca2#146](https://github.com/z0u/sca2/pull/146)). The patch above is applied by hand in `z0u-bot/sca2-backup`, so that copy of the workflow is ahead of the template until this lands.

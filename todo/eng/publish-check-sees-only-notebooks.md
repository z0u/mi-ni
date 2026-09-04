---
status: done
tags: [publishing]
opened: 2026-08-06
closed: 2026-08-13
---
# The publish check only sees changed notebooks

**Done via the directory**, which turned out to be the whole of the cheap half. `mini.reports.input_dir` names the files a report reads from the repo — the directory it already takes its export key from — and both staleness checks now consult it: `scripts/unpublished_reports.py` treats a change anywhere under that directory as changing the report (a sibling report excepted, being a second document rather than an input), and `export_reports.is_stale` takes its mtime from the newest thing in there rather than from the notebook alone, so `./go preview` re-exports on a re-run too. Deletes register via the directory's own mtime; `__pycache__` is skipped, so an import doesn't read as an edit.

Two corrections to the original note, both found while doing it. `docs/report.css` was never a leak: `set_report_styles` re-inlines it from source at build time, so editing it restyles every published report with no re-export (that's what the baked-in `css_file=` copy is for). And the check now derives its candidates from the reports that exist *now* rather than from the diff, so `--diff-filter=d` came out — a deleted report drops out on its own, while a deleted input still counts.

What's left is the half that can't be done locally: an input outside the report's own directory, i.e. a restyle in `src/mini/vis/*.py`. Split out as [shared inputs under `src/`](shared-inputs-under-src-are-invisible.md), since it needs store access and is a different piece of work.

Original note follows. `scripts/unpublished_reports.py` compares a git diff against `publish.lock`, so a report whose inputs moved while its `report.py` didn't — a re-run experiment writing new refs, a restyle in `src/mini/vis/*.py`, an edit to `docs/report.css` — passes the check while serving stale figures. A sibling `experiment.py` change is the cheap 80% heuristic (same directory, likely new results); the sharper version compares the store refs the last export resolved (`_assets/provenance.json` already records them) against what's current. Worth doing once we notice it biting.

## Notes

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/publish-check-sees-only-notebooks.md`](https://github.com/z0u/sca2/blob/main/todo/eng/publish-check-sees-only-notebooks.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on.

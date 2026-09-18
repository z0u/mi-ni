---
name: deps-routine
description: Weekly dependency review — security advisories, GitHub Action pin freshness, and upgrades worth taking. Usually run on a schedule.
model: opus
effort: medium
---

Check the project dependencies for currency: GitHub Actions, Python, and Node.

`./go deps` runs all three checks and filters the raw tool output down to something readable. It is read-only: nothing it does rewrites `uv.lock`.

```bash
./go deps              # all three
./go deps --audit      # advisories, from uv audit and npm audit
./go deps --actions    # Action pins against their newest upstream tag
./go deps --updates    # upgrades available to packages we declare
```

**If a check reports that a tool's output shape has changed, fix `scripts/deps.sh` as part of the run.** It is a thin wrapper over `uv audit`, `npm audit`, `git ls-remote` and `uv lock --dry-run`, and all of those move — `uv audit` is still marked experimental and its JSON schema says `preview`. A filter that has quietly stopped matching anything reports a clean bill of health, which is worse than no check at all. The reasoning behind each filter is in the script's comments.

For packages that have a security vulnerability, the default action is to upgrade it. If there's a problem upgrading it, consider whether the vulnerability is likely to affect this project in particular — we have no network-exposed surface, and we author the inputs we parse, so a class of advisory that matters for a web service often doesn't reach us. Say which way you judged it and why.

Note that `--audit` collapses duplicate advisory records before counting. OSV carries a GHSA and a PYSEC entry for the same underlying issue, so the raw number runs about twice the number of distinct problems. Report the distinct count.

For packages that we use a lot, also check if there are new features we would benefit from. For example, a new version of `ty` might have better support for `numpy`. `--updates` narrows the lock's upgrade set to the packages we named in `pyproject.toml`, which is a short enough list to read release notes for.

For GitHub Actions, `--actions` compares each `uses:` pin against the newest tag upstream. A floating major such as `@v7` tracks its own patches; a full version pin such as `@v10.0.1` does not, and nothing else will tell us when it falls behind — that is what [pin-refresh-policy.md](../../todo/eng/pin-refresh-policy.md) is about.

## What a run produces

Most weeks: nothing. No PR, no message. That is the successful outcome when there's nothing worth doing, and it's what keeps this different from a bot that opens a PR per stale pin.

Open a PR when you can give a reason in one sentence — an advisory we should clear, a fix or feature we'd feel, or a pin that has fallen a major behind. Plain staleness with nothing behind it doesn't qualify. Prefix the title `[DEP]`, and keep security upgrades in their own PR rather than folding them into a general refresh: they have a different urgency and want a different review.

Stay inside `pyproject.toml`, `uv.lock`, `package.json`, `package-lock.json`, and `.github/workflows/` — plus `scripts/deps.sh` when a filter needs repair. Run `./go check` before pushing.

If you find something that needs a decision rather than a change, add a note to the relevant `todo/eng/` item instead of opening a PR.

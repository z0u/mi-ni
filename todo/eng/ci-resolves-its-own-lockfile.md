---
status: done
tags: [ci, tooling]
opened: 2026-08-24
closed: 2026-08-25
---
# CI re-resolves the lockfile instead of failing on a stale one

`./go install` runs `uv sync --all-groups --no-group cuda`, with neither `--locked` nor `--frozen`, and CI runs `./go install`. Plain `uv sync` updates `uv.lock` when it disagrees with `pyproject.toml`, so a dependency edit pushed without a re-lock doesn't fail the build — CI quietly resolves its own lockfile, tests against that, and reports green. Whatever gets merged is a `pyproject.toml` the committed lock no longer describes, and the next person to run `uv sync` locally picks up a third resolution.

Confirmed rather than inferred, on 2026-08-24: bump one floor in `pyproject.toml` and leave the lock alone, and `uv lock --check` reports it stale and `uv sync --locked` errors with "The lockfile at `uv.lock` needs to be updated" — but plain `uv sync` installs the new version and rewrites `uv.lock` on the spot, no warning.

The relative cooldown is what makes this more than tidiness. `exclude-newer = "3 days"` resolves against wall-clock time at resolution, so a CI re-lock weeks after the pyproject edit sees a different eligible set than the developer who made it: newer releases, and a lockfile nobody reviewed. That is the same reproducibility claim the cooldown is there to make, arriving through the back door.

The fix is one flag, so the question is which. `--locked` fails the build and names the remedy, which reads right for CI and costs a red build on the first forgotten `uv lock` — the point. `--frozen` installs from the lock and ignores `pyproject.toml` entirely, which hides the drift rather than reporting it. But `install.sh` is shared with developers, where a hard failure on a half-finished dependency edit would be irritating, so it probably wants to be conditional on `$CI` (or a `--locked` flag on `./go install` that only the workflow passes) rather than unconditional.

The other two workflows have it too, and they don't go through `install.sh`: `pr-preview.yml:42` and `publish-docs.yml:41` both run a bare `uv sync --group pages`. So the fix is three edits, not one — and those two are CI-only, so they can take the flag unconditionally.

## Notes

**2026-08-25, closing** — Took `--locked`, defaulted from `$CI` in `install.sh`, with `--locked` / `--no-locked` to say so explicitly either way. The default reads `$CI` rather than being passed by the workflow, so a workflow added later inherits it instead of having to remember. Nothing in `lint-check.yml` says so: the reasoning sits in `install.sh` next to the line it governs, rather than being restated in every workflow that installs. The other two workflows take the flag unconditionally, as the body suggested, and each carries a short note, since there the flag is at the call site. The install step stays the gate every other check hangs off: a lockfile its manifest outgrew is an environment fault, which is the exception that step's comment already carves out, and uv's error names `uv lock` itself.

`npm install` had the same shape and the same script, so it went in the same change: `npm ci` under the locked default, which installs the lockfile as written and errors when `package.json` has outgrown it. Only `lint-staged` for the pre-commit hook rides on it, so the stake is small, but the failure mode was identical.

All four paths exercised before the push: clean tree passes under `CI=true`; a bumped floor in `pyproject.toml` with the lock untouched fails with "The lockfile at `uv.lock` needs to be updated" and leaves `uv.lock` alone; the same tree without `CI` still installs and re-resolves as before; and a `lint-staged` spec the lock can't satisfy fails `npm ci` by name.

**2026-09-04, backport** — Ported from sca2 ([`todo/eng/ci-resolves-its-own-lockfile.md`](https://github.com/z0u/sca2/blob/main/todo/eng/ci-resolves-its-own-lockfile.md) there) as settled: the fix it records is in this tree's code too, and the closing notes are the reasoning that code relies on. In this tree the two CI-only workflows have since folded into `site.yml`, which syncs no project environment at all (`uv run --no-project`), so only `lint-check.yml` is in scope, and it takes the `$CI` default through `./go install`.

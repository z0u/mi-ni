---
status: done
tags: [tooling, storage]
opened: 2026-09-05
closed: 2026-09-05
bundle: backup-template
---
# The backup template's `ruff.toml` extends a file the backup repo doesn't have

`templates/backup/ruff.toml` starts with `extend = "../../pyproject.toml"`, which resolves here, where the template sits two levels down. Installed, the file lands at the root of the backup repo, where that path points nowhere and ruff exits with an error before it reads `target-version`. Nothing in a backup repo runs ruff today, so it costs nothing until someone opens the repo in an editor with ruff configured, or adds a lint step; the `py312` pin the file exists to carry is the part that would then be missed.

The fix is to make the file self-contained — `target-version = "py312"` and whatever formatting settings the template wants — and to reword the comment, which currently says formatting is inherited.

## Notes

**2026-09-05, setup** — Found while installing the template into sca2's backup repo ([z0u/sca2#146](https://github.com/z0u/sca2/pull/146)). That install dropped the `extend` line by hand, so the two copies differ until this lands.

**2026-09-05, Opus** — Done. The file is self-contained now: the `py312` target, the line length, the Markdown exclusion, and the `select`/`ignore` pair the `extend` had been supplying, which is more than the item expected to find behind that one line. Linting `templates/backup/` from this repo is unchanged. A test copies the template somewhere with nothing above it and runs ruff there, which is the only place the flaw shows; it fails with the old config, for the reason above.

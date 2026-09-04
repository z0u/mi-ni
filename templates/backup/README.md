# Backup of <owner>/<project>

A nightly, pull-based copy of a mi-ni project: its code (the `mirror` branch and `snap/<date>` tags here), its Hugging Face store bucket, and its publish repo (under `store/` and `pub/` in the backup dataset). It runs from this repo so that no token the project's development environments hold can reach it, and it never deletes.

- [`.github/workflows/backup.yml`](./.github/workflows/backup.yml) — the schedule. The source and target names are its `env:` block.
- [`backup.py`](./backup.py) — the three legs.
- [`state/`](./state/) — `last-run.json`, what the last run saw and did.
- [`RESTORE.md`](./RESTORE.md) — how to get it all back.

Copied as-is from the mi-ni template's `templates/backup/`; the `backup` skill there is the setup runbook.

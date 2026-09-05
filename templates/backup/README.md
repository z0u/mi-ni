# Backup of <owner>/<project>

A nightly, pull-based copy of a mi-ni project: its code (the `mirror` branch and `snap/<date>` tags in the mirror repo), its Hugging Face store bucket (in the backup bucket, kept for a retention window after the source drops a file), and its publish repo (under `pub/` in the backup dataset, with the store's `refs/` history beside it). It runs from this repo so that no token the project's development environments hold can reach it, and it never deletes anything the source still has.

- [`.github/workflows/backup.yml`](./.github/workflows/backup.yml) — the schedule. The source and target names are its `env:` block, and its header comment says which secrets it takes.
- [`backup.py`](./backup.py) — the three legs.
- [`state/`](./state/) — `last-run.json`, what the last run saw and did; `store-missing.json`, the files the source has dropped and the date each was first missed.
- [`ruff.toml`](./ruff.toml) — pins the Python version `backup.py` is formatted for, so the recovery script keeps running on older interpreters.
- [`RESTORE.md`](./RESTORE.md) — how to get it all back.

Copied as-is from the mi-ni template's `templates/backup/`; the `backup` skill there is the setup runbook.

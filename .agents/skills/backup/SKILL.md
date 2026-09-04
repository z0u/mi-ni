---
name: backup
description: Nightly pull-based backup of a mi-ni project's GitHub repo, Hugging Face store bucket, and publish repo, installed from `templates/backup/` into a separate backup repo. Use to set it up, check that it is running, or restore.
---

# Backups the project's own tokens cannot reach

[`templates/backup/`](/templates/backup/) holds the files you install: a workflow, `backup.py` (which runs the three legs), a restore note, and a README. [`eng/environments.md`](/eng/environments.md) has the reasoning. This page is the runbook; the human's steps are marked.

A separate GitHub repo runs a nightly job. It pulls from the sources and writes into itself and into one Hugging Face dataset repo, using a token that can only read the sources and write to the backup. So no token held by a development environment can reach the backup. The job also never deletes anything, and it never runs code fetched from the sources.

Each leg lands somewhere different. The `mirror` branch tracks `main` on the source, and a `snap/<date>` tag records each night's tip. In the dataset, `store/` holds the files from the bucket; it is write-once by hash, so only new files move. And `pub/` replays the history of the publish repo one commit at a time, which keeps every revision a `publish.lock` has pinned recoverable.

## Setting it up

1. Pick the owning account (human). Sibling repos under the project's account are already out of reach of a leaked token, because fine-grained GitHub and HF tokens are scoped to named repos. Guarding against a compromised login, or a slip by the owner, needs a second account on both services: its own password and 2FA, recovery codes kept offline, and never logged in from the machines that hold the day-to-day tokens. Either choice leads to the same steps below. A backup of public sources can be public; a private source gets a private backup repo.

2. Create the backup GitHub repo (`<owner>/<project>-backup`), empty, and copy `templates/backup/` into it as-is (`cp -r templates/backup/. <backup-repo>/`). In the workflow, uncomment the `env:` block and fill in the four names; also fill in the title in the README. Commit to the default branch, since only the schedule on that branch runs.

3. Create the backup HF dataset repo, visibility as in step 1:

   ```bash
   uv run hf repos create <ns>/<project>-backup --type dataset
   ```

4. Mint the backup token (human): a fine-grained HF token with read on the store bucket and the publish repo, write on the backup dataset, and nothing else. Paste it into the Actions secrets of the backup repo as `HF_TOKEN`. No agent session should hold it.

5. Add rulesets on the backup repo (Settings → Rules → Rulesets): for all branches, block force pushes and restrict deletions; for all tags, restrict updates and deletions. A ruleset on a personal repo has no bypass list, so it binds the `GITHUB_TOKEN` of the workflow as well. Editing the ruleset needs admin access, which that token does not have. That is what makes a `snap/<date>` tag permanent.

6. Run it once (Actions → Backup → Run workflow), then check each leg against the sources. The `mirror` branch should sit at the tip of `main` on the source, and a `snap/<date>` tag should exist. The dataset should hold `store/cas/…`, `store/refs/…`, and `pub/…`, with `pub/SOURCE_COMMIT` naming the head commit of the publish repo. And `state/last-run.json` should have the counts and no `errors` key.

   Before any of that, a dry run can size the first copy using read access alone. It fetches the source into whatever `--repo` names, so point it at a scratch repo:

   ```bash
   git init -q /tmp/scratch
   SOURCE_REPO=… SOURCE_BUCKET=… SOURCE_PUBLISH_REPO=… BACKUP_DATASET=… \
     python templates/backup/backup.py --dry-run --repo /tmp/scratch --state /tmp/scratch/state.json
   ```

7. Protect the sources too. This is independent of the backup and nearly free. Protect `main` with a ruleset (or a branch rule with bypass off) that restricts deletions and blocks force pushes, and add a tag ruleset beside it. `./go auth` requests contents, issues, pull requests, and actions on the PAT, never administration, so a scoped PAT cannot lift such a rule. Keep HF tokens fine-grained and per-repo, and assume `repo.write` can delete a whole repo. Engineering environments hold a dev-pair token (see the `storage-envs` skill), which takes one production write token out of circulation.

## Keeping it running

GitHub disables a scheduled workflow in a public repo after 60 days without repository activity. The job commits `state/last-run.json` on every run, and that commit counts as the activity; a private repo is exempt anyway. If it ever stops, the `workflow_dispatch` trigger restarts it.

Two signals say the history of a source was rewritten, which is the incident the backup exists for. The run emits a `::warning::` when `main` on the source is no longer a descendant of the mirror. And it fails the `pub` leg when the marker there names a commit the history of the publish repo no longer contains. Inspect before doing anything else. The snapshots and the replayed commits already hold the record, and the other legs keep running.

Two flags handle scale. Both have defaults in `backup.py`, and you change them on the `python backup.py` line in the workflow. `--max-commits` (200) caps how much publish-repo history one night replays, leaving the rest for later nights. `--batch-bytes` (2 GiB) splits a large bucket copy into one commit per batch, so it fits on the disk of the runner.

## Restoring

[`RESTORE.md`](/templates/backup/RESTORE.md) in the backup repo covers the three legs. Each one restores on its own, and each needs only write access to the target. Once a quarter, restore into throwaway targets and compare. Seeding a dev pair from the backup (see the `storage-envs` skill) is the same drill, with something to show for it.

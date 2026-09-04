---
name: backup
description: Set up, verify, or restore the nightly pull-based backup of a mi-ni project's GitHub repo, Hugging Face store bucket, and publish repo, installed from `templates/backup/` into a separate backup repo. Use when creating the backup for a new project, checking that it is still running, or recovering a deleted or rewritten source.
---

# Backups the project's own tokens cannot reach

The payload is [`templates/backup/`](/templates/backup/): a workflow (the code leg, in plain git), a script (the two Hugging Face legs), a restore note, and a README. The reasoning is in [`eng/environments.md`](/eng/environments.md). This skill is the runbook, written for an agent; the steps that are the human's are marked.

The idea in one paragraph. A separate GitHub repo runs a nightly job that *pulls* from the sources and writes into itself and into one HF dataset repo, using a token with read on the sources and write on the backup only. No token a development environment holds can reach the backup repos; the job never deletes; and it never runs code fetched from the sources, since the script is the backup repo's own copy. The `mirror` branch tracks the source's `main`, a `snap/<date>` tag records each night's tip, `store/` in the dataset holds the bucket's files (write-once-by-hash, so only new files ever move), and `pub/` replays the publish repo's history one commit at a time, so every revision a `publish.lock` has ever pinned stays recoverable.

## Setting it up

1. Owning account (human). Sibling repos under the project's own account are outside the reach of a leaked *token*, since fine-grained GitHub and HF tokens are scoped to named repos. Protection from a compromised *login*, or the owner's own slip, needs a second account on both services: its own password and 2FA, recovery codes offline, never logged in from the machines that hold the day-to-day tokens. The steps below are the same either way. A backup of public sources can be public; a private source's leg gets a private backup repo.

2. Create the backup GitHub repo (`<owner>/<project>-backup`), empty. Copy the contents of `templates/backup/` into it as they are (`cp -r templates/backup/. <backup-repo>/`): the workflow is already at `.github/workflows/backup.yml`, and `state/` is the directory the run record lands in. Edit the four names in the workflow's `env:` block and the links in the README. Commit to the default branch: only the default branch's schedule runs.

3. Create the backup HF dataset repo (`<ns>/<project>-backup`), visibility as in step 1:

   ```bash
   uv run hf repos create <ns>/<project>-backup --type dataset
   ```

4. Mint the backup token (human). A fine-grained HF token with read on the store bucket and the publish repo, write on the backup dataset, nothing else. Paste it into the backup repo's Actions secrets as `HF_TOKEN`. No agent session should ever hold it.

5. Add a ruleset on the backup repo (Settings → Rules → Rulesets): one targeting all branches that blocks force pushes and restricts deletions, and one targeting all tags that restricts updates and deletions. A personal repo's ruleset has no bypass list, so the workflow's own `GITHUB_TOKEN` is bound too, and editing the ruleset needs admin access, which that token lacks. This is what makes a `snap/<date>` tag stay once pushed.

6. Run it once (Actions → Backup → Run workflow) and verify each leg against the sources: `mirror` is at the source's `main` tip, a `snap/<date>` tag exists, the dataset holds `store/cas/…` and `store/refs/…`, and `pub/SOURCE_COMMIT` names the publish repo's head commit. `state/last-run.json` records the counts and has no `errors` key. Before any of this, a dry run sizes the first copy with only read access:

   ```bash
   SOURCE_BUCKET=… SOURCE_PUBLISH_REPO=… BACKUP_DATASET=… python templates/backup/backup.py --dry-run --state /dev/null
   ```

7. Source-side prevention, independent of the backup and nearly free. `main`'s protection is a ruleset (or a branch rule with bypass off) that restricts deletions and blocks force pushes, with a tag ruleset beside it; `./go auth` requests contents, issues, pull requests, and actions on the PAT, never administration, so a scoped PAT cannot lift such a rule. HF tokens stay fine-grained and per-repo (assume `repo.write` can delete a whole repo). Engineering environments carry a dev-pair token, per the `storage-envs` skill, which takes one production write token out of circulation.

## Keeping it running

GitHub disables a scheduled workflow in a public repo after 60 days without repository activity. The job commits `state/last-run.json` on every run, and that commit is the activity, so the job keeps itself alive for as long as it runs; a private repo is exempt from the rule. If it ever stops, the `workflow_dispatch` trigger restarts it.

Two signals mean a source's history was rewritten, which is the incident the backup exists for. The workflow warns (`::warning::`) when the source's `main` is no longer a descendant of the mirror, and the script fails the `pub` leg when its marker names a commit the publish repo's history no longer contains. Inspect before doing anything else; the snapshots and the replayed commits already hold the record, and the other legs keep running.

Two knobs for scale, both defaulted in the workflow's `python backup.py` line: `--max-commits` (200) caps how much publish-repo history one night replays, with the rest following on later nights; `--batch-bytes` (2 GiB) splits a large bucket copy into one commit per batch so it fits the runner's disk.

## Restoring

[`RESTORE.md`](/templates/backup/RESTORE.md) in the backup repo covers the three legs, each on its own and each needing only write access to the target. Once a quarter, restore into throwaway targets and compare. Seeding a dev pair from the backup (the `storage-envs` skill) is that drill with a use.

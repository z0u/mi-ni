---
name: backup
description: Nightly pull-based backup of a mi-ni project's GitHub repo, Hugging Face store bucket, and publish repo, installed from `templates/backup/` into a separate backup repo. Use to set it up, check that it is running, or restore.
---

# Backups the project's own tokens cannot reach

[`templates/backup/`](/templates/backup/) holds the files you install: a workflow, `backup.py` (which runs the three legs), a restore note, and a README. [`eng/environments.md`](/eng/environments.md) has the reasoning. This page is the runbook; the human's steps are marked.

A separate GitHub repo, under a separate account, runs a nightly job. It pulls from the sources and writes into three targets that all belong to the backup account: a mirror repo on GitHub for the code, a dataset on the Hub for the publish history, and a bucket on the Hub for the store. So no token held by a development environment can reach the backup. The job never deletes anything the source still has, and it never runs code fetched from the sources.

No single token spans both accounts, so there are two kinds of credential. Reads use read-only tokens from the account that owns the sources, and only if a source is private; public sources are read anonymously. Writes use credentials from the backup account. The Hub targets take tokens minted fresh each run from their trusted publishers, so nothing long-lived with write access is stored for them. The mirror repo takes one stored fine-grained token, because the job's own token may not push workflow files.

Each leg lands somewhere different. In the mirror repo, the `mirror` branch tracks `main` on the source, and a `snap/<date>` tag records the tip each night. The backup bucket holds a copy of the store bucket, made server-side by content hash where it can be, so no bytes pass through the runner; a file the source drops stays there for a retention window (90 days) and is then let go. In the dataset, `pub/` replays the history of the publish repo one commit at a time, so every revision that a `publish.lock` has pinned stays recoverable, and `store/refs/` keeps every version of the store's name → artifact pointers, which the bucket, having no history, would forget on overwrite.

## Setting it up

1. Create the backup account (human). Fine-grained tokens are scoped to named repos, so a leaked token already can't reach sibling repos under the project's own account. A second account on both services covers more than that: a compromised login, or a slip by the owner. A plus-addressed email (`you+backup@…`) counts as a distinct account on both services and still lands in the inbox you already have. GitHub's terms allow one such machine account beside a personal one, for automated tasks only. Give it its own password and 2FA, keep the recovery codes offline, and never log in from the machines that hold the day-to-day tokens. Don't add the project's account as a collaborator on the backup repos, since collaborators on personal repos get write access. A backup of public sources can be public; a private source gets private backups.

2. Create the backup GitHub repo (`<owner>/<project>-backup`) under the backup account, and put the template in it. Copying from a development machine would mean holding a backup-account credential there, which is the adjacency the two accounts exist to avoid. A Codespace on the backup repo avoids it: its `GITHUB_TOKEN` is minted by GitHub and scoped to that repo, and `templates/backup/` can be fetched read-only over anonymous HTTPS, so the whole bootstrap stays one-directional in the same shape as the nightly job. Initialise the repo with a README so a Codespace can open it, then in its terminal:

   ```bash
   curl -sL https://github.com/<owner>/<project>/archive/refs/heads/main.tar.gz \
     | tar xz --strip-components=3 --wildcards '*/templates/backup'
   ```

   Uncomment the `env:` block and fill in the six names, fill in the title in the README, commit to the default branch (only the schedule on that branch runs), and delete the Codespace. Failing that, a fine-grained PAT with contents write on the one repo, used once and revoked, is the fallback. Add a ruleset here too (Settings → Rules → Rulesets): for all branches, block force pushes and restrict deletions, with an empty bypass list, so the job's own token cannot rewrite the history of the workflow that runs it.

3. Create the mirror repo (human, in the backup account): `<owner>/<project>-mirror`, empty, visibility as in step 1. Three settings make it safe to push a foreign history into, and the job checks the first before every push:

   - Settings → Actions → General → **Disable actions**. The source's commits carry its workflow files, and pushes made with a personal token trigger workflows, unlike pushes made with a job's own token. With Actions off, nothing pushed here can run.
   - Settings → Rules → Rulesets: for all branches, block force pushes and restrict deletions; for all tags, restrict updates and deletions. Leave the bypass list empty, so the rule binds the token below as well. That is what makes a `snap/<date>` tag permanent.
   - A fine-grained personal access token of the backup account, scoped to this one repo: Contents read and write, Workflows read and write, Administration read (that is how the job reads the Actions setting). Put it in the `MIRROR_GH_TOKEN` secret of the backup repo, and note its expiry; the leg fails cleanly when it lapses.

   This is the one stored write credential. It lives wholly in the backup account, reaches one repo that holds no secrets, and the rulesets leave it able only to append.

4. Create the Hub targets (human, in the backup account): the dataset `<ns>/<project>-backup` and the bucket `<ns>/<project>-backup-store`, visibility as in step 1. Leave the bucket in the default storage region, the same as the source bucket, since the server-side copy works only within one region. These have to be human steps, because an agent session holds the project's token, and that token can't create repos under another account.

5. Connect the write side (human, in the backup account). On the settings page of the dataset, and again on the bucket's, under Trusted Publishers, add GitHub Actions with repository `<owner>/<project>-backup`, branch `main`, and workflow `backup.yml`. Each run then exchanges the identity of the job for two tokens that last an hour and reach one target each; there is nothing to paste or rotate. If you'd rather store a token, put a fine-grained one with write on the dataset and the bucket, and nothing else, in the `HF_TOKEN` secret of the backup repo, and the workflow skips the minting steps.

   The exchange failed once, on a day the Hub was returning 500s, with `invalid_grant: This operation was aborted` while `repository`, `ref` and `workflow_ref` all matched the job; the same workflow minted fine the next day. That message is the wording of a cancelled request rather than of a claim mismatch, so treat it as a Hub-side hiccup: retry later. GitHub's immutable subject claim (`sub` as `repo:<owner>@<id>/<repo>@<id>:ref:…`, on every repo created after 2026-07-15) is accepted, so a new backup repo needs no opt-out. If a retry still fails with matching claims, report it to the Hub with the `Request ID` the CLI prints.

   Meanwhile the stored-token fallback keeps containment: the token is scoped to the two backup targets and lives wholly in the backup account, so what it costs is per-run rotation, not the separation the design is built on.

6. Add read tokens, if any source is private (human, in the project's account). The `SOURCE_HF_TOKEN` secret takes a fine-grained HF token with read on the bucket and the publish repo; a private bucket is read through it file by file, which the *Scale* section below sizes. The `SOURCE_GH_TOKEN` secret takes a fine-grained GitHub token with contents read on the source repo. Both are read-only, so a leak of the secrets in the backup repo can't delete anything, and revoking one only pauses the backup. Public sources need neither.

   Check visibility rather than assuming it, and check it the right way. The store bucket is a *bucket*, not a dataset, so `GET /api/datasets/<ns>/<name>` answers 401 for it however it is configured, and that 401 means "no such dataset" rather than "private". Ask `HfApi().bucket_info(...)`, or read it anonymously with `HfApi(token=False)`. Watch the credential too: `HF_TOKEN` is often unset in a devcontainer shell, the token sitting in `~/.cache/huggingface/token` where the client finds it on its own, so a hand-rolled `curl -H "Authorization: Bearer $HF_TOKEN"` sends an empty credential and 401s for that reason instead.

7. Run it once (Actions → Backup → Run workflow), then check each leg against the sources. The mirror repo's `mirror` branch should sit at the tip of `main` on the source, and a `snap/<date>` tag should exist. The backup bucket should hold `cas/…` and `refs/…`, with the same file count as the source. The dataset should hold `store/refs/…` and `pub/…`, with `pub/SOURCE_COMMIT` naming the head commit of the publish repo. And `state/last-run.json` should have the counts and no `errors` key.

   Before any of that, a dry run can size the first copy. It needs no write token, and no token at all for public sources (pass `SOURCE_HF_TOKEN=…` for a private one). It fetches the source into whatever `--repo` names, so point that at a scratch repo:

   ```bash
   git init -q /tmp/scratch
   SOURCE_REPO=… SOURCE_BUCKET=… SOURCE_PUBLISH_REPO=… \
   MIRROR_REPO=… BACKUP_DATASET=… BACKUP_BUCKET=… \
     python templates/backup/backup.py --dry-run --repo /tmp/scratch --state /tmp/scratch/state.json
   ```

8. Protect the sources too. This is independent of the backup and nearly free. Protect `main` with a ruleset (or a branch rule with bypass off) that restricts deletions and blocks force pushes, and add a tag ruleset beside it. `./go auth` requests contents, issues, pull requests, and actions on the PAT, never administration, so a scoped PAT cannot lift such a rule. Keep HF tokens fine-grained and per-repo, and assume `repo.write` can delete a whole repo. Engineering environments hold a dev-pair token instead (see the `storage-envs` skill), which takes one production write token out of circulation.

## Why the code goes to a second repo

`GITHUB_TOKEN` may not create or update files under `.github/workflows/`, and no permission grants it, so a mirror pushed into the backup repo itself is rejected for any source that has workflows: `refusing to allow a GitHub App to create or update workflow ... without 'workflows' permission`. Excluding or renaming those files is not a way out. They sit inside the commit trees, so dropping them means rewriting every commit, and rewritten shas cost the things that make this a backup: the snapshot no longer records the source's real tip, the descendant check compares against a history that exists nowhere else, and a restore returns a repo that is not the one that was lost.

A personal token with workflows write can push them, and that is the token the leg uses. But a push made with a personal token triggers workflows, and the commits being pushed carry the source's. So the mirror lives in a repo of its own with Actions switched off, where nothing pushed can run, and the leg reads that setting through the API before each push and refuses while it is on. Disabling workflows one by one through the API after the push, which is sometimes suggested, does not close the gap: the API only knows workflow files that have already reached a pushed ref, so a new one would run once first.

## Keeping it running

GitHub disables a scheduled workflow in a public repo after 60 days without repository activity. The job commits `state/` on every run, and that commit counts as the activity; a private repo is exempt anyway. If it ever stops, the `workflow_dispatch` trigger restarts it. The minted write tokens need no rotation. The mirror token and a stored read token, if there is one, are the secrets with an expiry; the code leg fails on its own, with the other two unaffected, when the mirror token lapses.

Two signals say the history of a source was rewritten, which is the incident the backup exists for. The run emits a `::warning::` when `main` on the source is no longer a descendant of the mirror. And the `pub` leg fails when the marker there names a commit that the history of the publish repo no longer contains. Inspect before doing anything else. The snapshots and the replayed commits already hold the record, and the other legs keep running.

The store leg follows the source with a delay, and the run record says how far behind it is. `missing` in `state/last-run.json` counts the files the backup bucket holds that the source no longer does, and `state/store-missing.json` names them with the date each was first missed; `expired` counts the ones deleted that night, their window having passed. A `mini gc --store` upstream shows up as a batch of newly missing files and nothing else. If more than half of the backed-up files vanish from the source in one night, the run emits a `::warning::` and starts their clocks as usual, so there is the whole window to look. Nothing shortens a clock: the source can start one by deleting a file, and a lost or reset manifest only delays every expiry.

Two flags handle scale, and one the window. All have defaults in `backup.py`, and you change them on the `python backup.py` line in the workflow. `--max-commits` (200) caps how much publish-repo history one night replays, leaving the rest for later nights. `--retain-days` (90) is how long the backup bucket keeps a file after the source drops it; longer than the 14-day grace of `mini gc --store` by design, so a blob swept by mistake is still recoverable a season later. The store copy usually moves no bytes through the runner, since the Hub copies bucket files server-side by content hash, so a large store costs a listing and a few API calls. That copy is one request, authorised against the backup bucket's token alone, and a fine-grained token carries implicit read on public repos only — so a *private* source bucket is invisible to it. The leg asks first (`bucket_info`) and falls back to downloading and re-uploading each file, in batches bounded by `STAGE_BYTES` (2 GB) so the runner's disk holds one batch rather than the whole delta; `state/last-run.json` says when a run took that route. It is the first run that costs, since the delta is small afterwards, and it keeps the property that no token spans both accounts. Making the source bucket public, if nothing in it argues against that, restores the server-side copy. The `pub` leg pays a round trip per commit however small the commit is (a first replay of 57 commits took five and a half minutes), so `--max-commits` is the flag that matters for a long history. Reads of a public source are anonymous by design, which the Hub warns about; if a first replay drags, a read-only `SOURCE_HF_TOKEN` buys higher rate limits without making any stored secret writable.

## Restoring

[`RESTORE.md`](/templates/backup/RESTORE.md) in the backup repo covers the three legs. Each one restores on its own, and each needs only write access to the target; the store comes back server-side, in seconds. Once a quarter, restore into throwaway targets and compare, and refresh a copy on a machine of your own. Seeding a dev pair from the backup (see the `storage-envs` skill) runs through the same steps and leaves you with something usable.

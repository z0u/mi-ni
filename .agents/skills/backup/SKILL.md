---
name: backup
description: Nightly pull-based backup of a mi-ni project's GitHub repo, Hugging Face store bucket, and publish repo, installed from `templates/backup/` into a separate backup repo. Use to set it up, check that it is running, or restore.
---

# Backups the project's own tokens cannot reach

[`templates/backup/`](/templates/backup/) holds the files you install: a workflow, `backup.py` (which runs the three legs), a restore note, and a README. [`eng/environments.md`](/eng/environments.md) has the reasoning. This page is the runbook; the human's steps are marked.

A separate GitHub repo, under a separate account, runs a nightly job. It pulls from the sources, then writes into itself and into one Hugging Face dataset repo. So no token held by a development environment can reach the backup. The job never deletes anything, and it never runs code fetched from the sources.

No single token spans both accounts, so there are two kinds of credential. Reads use read-only tokens from the account that owns the sources, and only if a source is private; public sources are read anonymously. Writes use a credential from the backup account. By default that is a token minted fresh each run from the trusted publisher of the dataset, so nothing long-lived with write access is stored anywhere.

Each leg lands somewhere different. The `mirror` branch tracks `main` on the source, and a `snap/<date>` tag records the tip each night. In the dataset, `store/` holds the files from the bucket. It is write-once by hash, so only new files move. And `pub/` replays the history of the publish repo one commit at a time, so every revision that a `publish.lock` has pinned stays recoverable.

## Setting it up

1. Create the backup account (human). Fine-grained tokens are scoped to named repos, so a leaked token already can't reach sibling repos under the project's own account. A second account on both services covers more than that: a compromised login, or a slip by the owner. A plus-addressed email (`you+backup@…`) counts as a distinct account on both services and still lands in the inbox you already have. GitHub's terms allow one such machine account beside a personal one, for automated tasks only. Give it its own password and 2FA, keep the recovery codes offline, and never log in from the machines that hold the day-to-day tokens. Don't add the project's account as a collaborator on the backup repo, since collaborators on personal repos get write access. A backup of public sources can be public; a private source gets a private backup repo.

2. Create the backup GitHub repo (`<owner>/<project>-backup`) under the backup account, and put the template in it. Copying from a development machine would mean holding a backup-account credential there, which is the adjacency the two accounts exist to avoid. A Codespace on the backup repo avoids it: its `GITHUB_TOKEN` is minted by GitHub and scoped to that repo, and `templates/backup/` can be fetched read-only over anonymous HTTPS, so the whole bootstrap stays one-directional in the same shape as the nightly job. Initialise the repo with a README so a Codespace can open it, then in its terminal:

   ```bash
   curl -sL https://github.com/<owner>/<project>/archive/refs/heads/main.tar.gz \
     | tar xz --strip-components=3 --wildcards '*/templates/backup'
   ```

   Uncomment the `env:` block and fill in the four names, fill in the title in the README, commit to the default branch (only the schedule on that branch runs), and delete the Codespace. Failing that, a fine-grained PAT with contents write on the one repo, used once and revoked, is the fallback — add the rulesets of step 6 first, so a leaked bootstrap token could only append.

3. Create the backup HF dataset repo (human, in the backup account): `<ns>/<project>-backup`, visibility as in step 1. This one has to be a human step, because an agent session holds the project's token, and that token can't create repos under another account.

4. Connect the write side (human, in the backup account). On the settings page of the dataset, under Trusted Publishers, add GitHub Actions with repository `<owner>/<project>-backup`, branch `main`, and workflow `backup.yml`. Each run then exchanges the identity of the job for a token that lasts an hour and reaches that one dataset; there is nothing to paste or rotate. If you'd rather store a token, put a fine-grained one with write on the dataset only in the `HF_TOKEN` secret of the backup repo, and the workflow skips the minting step.

   The exchange can fail with `invalid_grant: This operation was aborted` while `repository`, `ref` and `workflow_ref` all match the job exactly. A repo created after 2026-07-15 carries GitHub's [immutable subject claim](https://github.blog/changelog/2026-04-23-immutable-subject-claims-for-github-actions-oidc-tokens/), whose `sub` embeds the owner and repo ids — `repo:<owner>@<id>/<repo>@<id>:ref:…` rather than `repo:<owner>/<repo>:ref:…` — and it is not clear that the Hub matches such a token. The repo's `use_immutable_subject` setting is the opt-in for older repos, not an opt-out for new ones. To see what a run actually presents, decode the payload of its id token (never the token itself):

   ```yaml
   - run: |
       curl -sS -H "Authorization: Bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
         "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=https://huggingface.co" \
       | python3 -c 'import sys,json,base64
   t = json.load(sys.stdin)["value"].split(".")[1]
   c = json.loads(base64.urlsafe_b64decode(t + "=" * (-len(t) % 4)))
   print({k: c.get(k) for k in ("iss","aud","repository","ref","workflow_ref","sub")})'
   ```

   Take the stored-token fallback rather than fighting it. The fallback keeps containment: the token is scoped to the one backup dataset and lives wholly in the backup account, so what it costs is per-run rotation, not the separation the design is built on.

5. Add read tokens, if any source is private (human, in the project's account). The `SOURCE_HF_TOKEN` secret takes a fine-grained HF token with read on the bucket and the publish repo. The `SOURCE_GH_TOKEN` secret takes a fine-grained GitHub token with contents read on the source repo. Both are read-only, so a leak of the secrets in the backup repo can't delete anything, and revoking one only pauses the backup. Public sources need neither.

   Check visibility rather than assuming it, and check it the right way. The store bucket is a *bucket*, not a dataset, so `GET /api/datasets/<ns>/<name>` answers 401 for it however it is configured, and that 401 means "no such dataset" rather than "private". Ask `HfApi().bucket_info(...)`, or read it anonymously with `HfApi(token=False)`. Watch the credential too: `HF_TOKEN` is often unset in a devcontainer shell, the token sitting in `~/.cache/huggingface/token` where the client finds it on its own, so a hand-rolled `curl -H "Authorization: Bearer $HF_TOKEN"` sends an empty credential and 401s for that reason instead.

6. Add rulesets on the backup repo (Settings → Rules → Rulesets): for all branches, block force pushes and restrict deletions; for all tags, restrict updates and deletions. Leave the bypass list empty, so the rule binds the `GITHUB_TOKEN` of the workflow as well. Editing the ruleset needs admin access, which that token does not have. That is what makes a `snap/<date>` tag permanent.

7. Run it once (Actions → Backup → Run workflow), then check each leg against the sources. The `mirror` branch should sit at the tip of `main` on the source, and a `snap/<date>` tag should exist. The dataset should hold `store/cas/…`, `store/refs/…`, and `pub/…`, with `pub/SOURCE_COMMIT` naming the head commit of the publish repo. And `state/last-run.json` should have the counts and no `errors` key.

   Before any of that, a dry run can size the first copy. It needs no write token, and no token at all for public sources (pass `SOURCE_HF_TOKEN=…` for a private one). It fetches the source into whatever `--repo` names, so point that at a scratch repo:

   ```bash
   git init -q /tmp/scratch
   SOURCE_REPO=… SOURCE_BUCKET=… SOURCE_PUBLISH_REPO=… BACKUP_DATASET=… \
     python templates/backup/backup.py --dry-run --repo /tmp/scratch --state /tmp/scratch/state.json
   ```

8. Protect the sources too. This is independent of the backup and nearly free. Protect `main` with a ruleset (or a branch rule with bypass off) that restricts deletions and blocks force pushes, and add a tag ruleset beside it. `./go auth` requests contents, issues, pull requests, and actions on the PAT, never administration, so a scoped PAT cannot lift such a rule. Keep HF tokens fine-grained and per-repo, and assume `repo.write` can delete a whole repo. Engineering environments hold a dev-pair token instead (see the `storage-envs` skill), which takes one production write token out of circulation.

## A gap in the code leg

`GITHUB_TOKEN` may not create or update files under `.github/workflows/`, and no `workflows` scope exists to grant it, so the code leg's push is rejected for any source repo that has workflows: `refusing to allow a GitHub App to create or update workflow ... without 'workflows' permission`. Neither the `mirror` branch nor the `snap` tag lands. The store and pub legs are unaffected, and the run records the failure against `code` alone.

Excluding or renaming the workflow files is not the way out. They sit inside the commit trees, so dropping them means rewriting every commit, and rewritten shas cost the things that make this a backup: the snapshot no longer records the source's real tip, the descendant check that detects an upstream history rewrite compares against a history that exists nowhere else, and a restore returns a repo that is not the one that was lost.

Two real options. Give the push a fine-grained PAT with contents and workflows write on the backup repo, which keeps the mirror browsable on github.com at the cost of a stored secret; or ship the leg as a `git bundle` into the backup dataset, which sidesteps a rule that only governs pushes, keeps every ref and sha, needs no GitHub secret at all, and gives up browsing for `git clone backup.bundle`.

## Keeping it running

GitHub disables a scheduled workflow in a public repo after 60 days without repository activity. The job commits `state/last-run.json` on every run, and that commit counts as the activity; a private repo is exempt anyway. If it ever stops, the `workflow_dispatch` trigger restarts it. The minted write token needs no rotation. A stored read token, if there is one, is the only secret with an expiry.

Two signals say the history of a source was rewritten, which is the incident the backup exists for. The run emits a `::warning::` when `main` on the source is no longer a descendant of the mirror. And the `pub` leg fails when the marker there names a commit that the history of the publish repo no longer contains. Inspect before doing anything else. The snapshots and the replayed commits already hold the record, and the other legs keep running.

Two flags handle scale. Both have defaults in `backup.py`, and you change them on the `python backup.py` line in the workflow. `--max-commits` (200) caps how much publish-repo history one night replays, leaving the rest for later nights. `--batch-bytes` (2 GiB) splits a large bucket copy into one commit per batch, so it fits on the disk of the runner.

Both legs move their bytes through the runner — download, then upload — since the Hub's server-side cross-repo copy covers repos, and a bucket is not one. Bytes are rarely what costs, though. A first run of 152 files took about forty-five seconds, while replaying 57 publish commits took five and a half minutes, because that leg pays a round trip per commit however small the commit is. So `--max-commits` is the flag that matters for a long history, and `--batch-bytes` for a large bucket. Reads of a public source are anonymous by design, which the Hub warns about; if a first copy drags, a read-only `SOURCE_HF_TOKEN` buys higher rate limits without making any stored secret writable.

## Restoring

[`RESTORE.md`](/templates/backup/RESTORE.md) in the backup repo covers the three legs. Each one restores on its own, and each needs only write access to the target. Once a quarter, restore into throwaway targets and compare, and refresh a copy on a machine of your own. Seeding a dev pair from the backup (see the `storage-envs` skill) runs through the same steps and leaves you with something usable.

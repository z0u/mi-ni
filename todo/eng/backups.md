---
status: open
tags: [archival, versioning, security, needs-design, publishing, storage]
opened: 2026-08-12
bundle: env-hardening
---

# Create hard-to-delete backups of code and experiment data

We have a few environments with write-enabled GitHub and HF tokens. If an attacker gained access to any of those environments, it could all go up in smoke. `main` has some branch protection, but I'm unsure how strong it is in practice, since the GH tokens are mostly issued for a principal who owns the repo. And the experiment data in HF is certainly deletable with the current tokens. 🚨 Obviously, don't try to test whether `main` and the HF data can be deleted.

Configure automatic indelible backups. This should be done in such a way that a _write_ token scoped to this GH repo or the HF bucket or HF dataset would be unable to delete the backups. Specifically:

- GH code `z0u/mi-ni`
- HF bucket `z0u/mi-ni-store`
- HF publish repo `z0u/mi-ni-pub`

## Proposed design (for review, not yet built)

**Two threat levels, and which one you are buying protection from.** The first is a leaked *token*: whatever a development environment holds (a fine-grained GitHub PAT scoped to `z0u/mi-ni`, the Claude GitHub App's installation token, a fine-grained HF token with `repo.write` on the bucket and the publish repo, a Modal token). The second is a compromised *login*, or the owner's own mistake: anything the account can do, the attacker can do, including deleting a repo under that account. Everything below defends against the first level with sibling repos under the same account, because the tokens in question are scoped to named repos (this session's HF token: `repo.write` on the bucket and publish repo, no user-level permissions), so a sibling repo is already outside their reach. Defending against the second level needs a **second account**, on GitHub and on Hugging Face, that owns the backup repos: separate password and 2FA, recovery codes kept offline, never logged in from the machines that hold the day-to-day tokens. The design is the same either way; the only difference is who owns the two backup repos, so the template need not care. For this repo (public template, demo data) sibling repos are proportionate; for a project whose data would hurt to lose, the second account is the one that protects the data from its owner. Even then the ceiling is "both accounts compromised at once": neither GitHub nor HF can stop an owner deleting their own repo. Past that ceiling lies object lock (S3 or B2 in compliance mode, where even the account root cannot delete before retention expires), noted under *Not chosen* below.

**Shape: pull-based mirroring from a separate trust domain.** A backup GitHub repo (`z0u/mi-ni-backup`) runs a nightly GitHub Actions workflow. It fetches the source repo and pushes into itself with its own `GITHUB_TOKEN` (scoped to the backup repo alone, so nothing is stored). Its one secret is an HF token with read on `z0u/mi-ni-store` and `z0u/mi-ni-pub` and write on an HF *dataset* repo (`z0u/mi-ni-backup`), a dataset rather than a bucket because the backup wants git history. That token is created for the backup and pasted only into the backup repo's Actions secrets, where only the owning account can read it.

**Visibility follows the source.** A backup of public sources is public: the GitHub repo and the HF dataset both. Public costs nothing in exposure (the snapshot tags and replayed commits show what the sources already show), lifts the private-repo Actions minute cap, and means a restore or a spot-check needs no token at all. Where a source is private, as the store bucket is in the projects built from this template, that leg's backup repo is private; the template makes this a per-leg setting.

**Keeping the schedule alive.** GitHub disables a scheduled workflow in a *public* repo after 60 days without repository activity, and only the default branch's schedule runs. The job commits a small `state/last-run.json` every run (timestamp, the source shas and file counts it saw), so each nightly run is itself repository activity; the file doubles as the run log and as the marker the delta walk reads. Whether a commit pushed by `GITHUB_TOKEN` counts as activity is not stated in GitHub's docs, and I could not confirm it from a primary source; the evidence that it does is that the keepalive actions people rely on for exactly this (efrecon/gh-action-keepalive, for one) work by committing a marker file from inside the scheduled workflow, which is the same mechanism. Two belts, neither needing a cross-repo credential: a private backup repo is exempt from the rule altogether (the free minute cap is ample for a daily two-minute job), and the restore drill below is the periodic check that the job is still running. A `workflow_dispatch` trigger stays on the workflow for a manual run after any pause.

**What the job does, three legs.**

- *Code.* Fetch `main` from the source (tags too, if wanted; they are cheap and already immutable pointers), update the backup's mirror of it, and create a tag `snap/<date>` whenever the tip moved. Tags share objects with the mirror, so a snapshot costs bytes only for new commits. A force-push upstream moves the mirror but never the snapshot. A ruleset on the backup repo restricts tag deletion and updates and blocks force pushes on branches; a personal repo's ruleset has no bypass list, so the backup's own workflow token is bound by it too, and editing the ruleset needs admin access, which `GITHUB_TOKEN` lacks.
- *Store bucket.* List the bucket, diff against the backup dataset's tree, and upload only files the backup lacks under `store/`, plus `store/refs/` overwritten each run. One commit per run. The CAS is write-once-by-hash, so "missing" is the whole delta, and a blob later overwritten in place upstream leaves the backup's original copy untouched. Today the bucket is 3.6 MB (100 CAS blobs), so the first run is trivial; at 100 GB the same delta walk holds because only new blobs move.
- *Publish repo.* It is git-backed, so replay its history: for each source commit the backup has not seen (oldest first), `snapshot_download` at that revision and `upload_folder` into the backup under `pub/`, with the source sha in the commit message. Xet dedup makes each replayed commit a metadata-sized upload. That keeps every revision a `publish.lock` pin has ever named recoverable, which a plain snapshot of `main` would lose.

**Two rules that keep the backup trustworthy.** The backup never deletes anything, and it never executes code fetched from the source. The script and workflow live in the backup repo, copied from `templates/backup/` in this repo. A script pulled from the mirrored source at head would let an attacker with write on `mi-ni` rewrite the backup job itself.

**Restore, and proving it works.** A `RESTORE.md` in the backup repo covers the three legs: `git push` from the mirror, `upload_folder(store/)` into a fresh bucket, and replaying `pub/` into a fresh dataset repo. A quarterly drill restores into throwaway repos and diffs; the non-prod design in `non-prod.md` can use the same restore to seed a dev bucket, which makes the drill a routine step rather than a ceremony.

**Prevention on the source side, independent of the backup and nearly free.** (1) Check that `main`'s protection is a ruleset (or a branch protection rule with bypass disabled) with restrict-deletions and block-force-pushes, and add one for tags. `./go auth` requests contents, issues, pull requests, and actions on the PAT, never administration, so a scoped PAT cannot lift such a rule. (2) Keep HF tokens fine-grained and per-repo, as they are; whether `repo.write` can delete a whole repo is not stated in HF's docs, so assume it can. (3) Science environments need production write tokens to run experiments, and CI carries a read-only one; an environment set aside for engineering work can carry a token scoped to the dev pair from `non-prod.md`, which takes one production write token out of circulation.

**Setup as a skill, payload as a template.** The runbook lives in this repo as `.agents/skills/backup/SKILL.md`, written for an agent: create the backup repo from `templates/backup/` (workflow, script, `RESTORE.md`), create the HF dataset, add the ruleset, run the workflow once, verify the three legs against the sources, and the restore procedure. It says plainly which steps are the human's: choosing the owning account, minting the backup token and pasting it into the backup repo's secrets, so no agent session ever holds it. The `skills-agents-reorg.md` item may move where skills live (a plugin); the backup skill goes wherever that lands.

**Not chosen.** A Modal cron in the existing workspace: the development tokens reach that workspace, so the job and its secret would be inside the blast radius. HF scheduled Jobs: viable and provider-native, but pay-as-you-go and not needed at this size. An object-locked S3 or B2 bucket is the only true "indelible" primitive; worth adding as a fourth leg (a weekly `git bundle` plus store tarball) if the data ever becomes irreplaceable, and it is a second provider to manage, so I have left it out of the first cut.

**Cost and effort.** No money: a public repo's Actions minutes are free, and a dataset of this size sits well under HF's quota. About one workflow file, a script of roughly 150 lines, a ruleset, and a restore note.

## Notes

**2026-09-01, z0u** — This sounds complicated. Investigate, design, maybe prototype, but let me review the design before building.

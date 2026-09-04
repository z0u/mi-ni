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

**Threat model.** The attacker holds whatever a development environment holds: a fine-grained GitHub PAT scoped to `z0u/mi-ni`, the Claude GitHub App's installation token, a fine-grained HF token with `repo.write` on the bucket and the publish repo, and a Modal token. They do not hold Sandy's account logins. Everything below follows from one consequence of that model: the backup must be reachable by none of those tokens, so it lives in repos those tokens are not scoped to, written by a credential that exists in exactly one place the development environments cannot read.

**Shape: pull-based mirroring from a separate trust domain.** A *vault* GitHub repo (`z0u/mi-ni-vault`) runs a nightly GitHub Actions workflow. It fetches the source repo and pushes into itself with its own `GITHUB_TOKEN` (scoped to the vault alone, so nothing is stored). Its one secret is an HF token with read on `z0u/mi-ni-store` and `z0u/mi-ni-pub` and write on an HF *dataset* repo (`z0u/mi-ni-vault`), chosen over a bucket because the vault wants git history. That token is created for the vault and pasted only into the vault repo's Actions secrets. The fine-grained tokens in development environments are scoped to named repos (this session's HF token: `repo.write` on the bucket and publish repo, no user-level permissions), so sibling repos under the same account are already outside their reach. A separate GitHub and HF account for the vault removes the remaining dependence on every environment token being scoped, at the cost of a second login; my recommendation is to start with sibling repos and move the vault to its own account if a broad token ever has to be issued.

**Visibility follows the source.** A vault for public sources is public: the GitHub repo and the HF dataset both. Public costs nothing in exposure (the snapshot tags and replayed commits show what the sources already show), lifts the private-repo Actions minute cap, and means a restore or a spot-check needs no token at all. Where a source is private, as sca2's bucket is, that leg's vault repo is private; the template makes this a per-leg setting rather than a global one.

**Keeping the schedule alive.** GitHub disables a scheduled workflow in a public repo after 60 days without repository activity, and only the default branch's schedule runs. The job therefore commits a small `state/last-run.json` every run (timestamp, the source shas and file counts it saw), so each nightly run is itself repository activity; the file doubles as the run log and as the marker the delta walk reads. A `workflow_dispatch` trigger stays on the workflow for a manual run after any pause, and the restore drill below is the periodic check that the job is still running.

**What the job does, three legs.**

- *Code.* Fetch every branch and tag from the source into the vault (forced, so the mirror tracks the source), then create a tag `snap/<date>/<branch>` for each branch whose tip moved. Tags share objects with the mirror, so a snapshot costs bytes only for new commits. A force-push or branch deletion upstream moves the mirror but never the snapshot. A ruleset on the vault repo restricts deletions and updates on tags and blocks force pushes on branches; a personal repo's ruleset has no bypass list, so the vault's own workflow token is bound by it too, and editing the ruleset needs admin access, which `GITHUB_TOKEN` lacks.
- *Store bucket.* List the bucket, diff against the vault dataset's tree, and upload only files the vault lacks under `store/`, plus `store/refs/` overwritten each run. One commit per run. The CAS is write-once-by-hash, so "missing" is the whole delta, and a blob later overwritten in place upstream leaves the vault's original copy untouched. Today the bucket is 3.6 MB (100 CAS blobs), so the first run is trivial; at 100 GB the same delta walk holds because only new blobs move.
- *Publish repo.* It is git-backed, so replay its history: for each source commit the vault has not seen (oldest first), `snapshot_download` at that revision and `upload_folder` into the vault under `pub/`, with the source sha in the vault commit message. Xet dedup makes each replayed commit a metadata-sized upload. That keeps every revision a `publish.lock` pin has ever named recoverable, which a plain snapshot of `main` would lose.

**Two rules that keep the vault trustworthy.** The vault never deletes anything, and the vault never executes code fetched from the source. The backup script and workflow live in the vault repo, copied from a `templates/vault/` directory in this repo (the template-user story: copy the folder, create two repos, paste one token). A script pulled from the mirrored source at head would let an attacker with write on `mi-ni` rewrite the backup job itself.

**Restore, and proving it works.** A `RESTORE.md` in the vault covers the three legs: `git push` from the mirror, `upload_folder(store/)` into a fresh bucket, and replaying `pub/` into a fresh dataset repo. A quarterly drill restores into throwaway repos and diffs; the non-prod design in `non-prod.md` can use the same restore to seed a dev bucket, which makes the drill a routine step rather than a ceremony.

**Prevention on the source side, independent of the vault and nearly free.** (1) Check that `main`'s protection is a ruleset (or a branch protection rule with bypass disabled) with restrict-deletions and block-force-pushes, and add one for tags. `./go auth` requests contents, issues, pull requests, and actions on the PAT, never administration, so a scoped PAT cannot lift such a rule. (2) Keep HF tokens fine-grained and per-repo, as they are; whether `repo.write` can delete a whole repo is not stated in HF's docs, so assume it can. (3) This Claude Code web session's HF token carries `repo.write` on both production stores, where `tests/mini/test_hf_store.py` says web sessions were meant to carry read-only tokens. Worth reconciling one way or the other.

**Setup as a skill, payload as a template.** The runbook lives in this repo as `.agents/skills/vault/SKILL.md`, written for an agent: create the vault repo from `templates/vault/` (workflow, script, `RESTORE.md`), create the HF dataset, add the ruleset, run the workflow once, verify the three legs against the sources, and the restore procedure. It says plainly which steps are the human's: minting the vault token and pasting it into the vault's secrets, so no agent session ever holds it. The `skills-agents-reorg.md` item may move where skills live (a plugin); the vault skill goes wherever that lands.

**Not chosen.** A Modal cron in the existing workspace: the development tokens reach that workspace, so the job and its secret would be inside the blast radius. HF scheduled Jobs: viable and provider-native, but pay-as-you-go and not needed at this size. An object-locked S3 or B2 bucket is the only true "indelible" primitive (compliance-mode retention binds even the account owner); worth adding as a fourth leg (a weekly `git bundle` plus store tarball) if the data ever becomes irreplaceable, and it is a second provider to manage, so I have left it out of the first cut.

**Cost and effort.** No money: a private repo's free Actions minutes cover a daily two-minute job, and a private dataset of this size sits well under HF's quota. About one workflow file, a script of roughly 150 lines, a ruleset, and a restore note.

## Notes

**2026-09-01, z0u** — This sounds complicated. Investigate, design, maybe prototype, but let me review the design before building.

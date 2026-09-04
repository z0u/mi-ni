# Environments: production, dev, and the backup

*Part of the [engineering notes](./README.md).*

Three places a project's bytes live, and why each is where it is. The *how* is in two skills: `storage-envs` (the dev pair) and `backup` (the backup repos).

## The profile picks names; the token draws the boundary

A `[tool.mini.profiles.<name>]` table, selected by `MINI_PROFILE`, replaces the two storage keys (`store-bucket`, `publish-repo`) and inherits the rest. Two decisions in that sentence.

*Replaces, never merges the pair.* A profile that names only a bucket has no publish repo, rather than production's. Inheriting the missing key would send a dev publish to the production repo the first time someone wrote a half profile, and the failure would look like success. Falling back to "unset" is what a project without the key gets anyway (a local store, or single-bucket publishing), so the safe behaviour is also the ordinary one.

*Inherits everything else.* The first cut replaced the whole table, and `MINI_PROFILE=dev` then dropped `app`, `env`, and `region`, so the CLI forgot its backend. Those keys describe the compute, which a sandbox shares with production; only the two storage names are what a sandbox exists to change.

The profile is a convenience, though. The boundary is the credential: an engineering environment holds a token with write on the dev pair only, so a session that forgets the profile fails on its first write. An environment that configures storage by variable (a Claude Code web environment sets `MINI_STORE_BUCKET` and `MINI_PUBLISH_REPO`, and has no `mini.local.toml`) needs no profile at all: point the two variables at the dev pair, with the dev token beside them.

There is no promotion step, by design. Science runs and their reports always use production; the publish tier already stages them, since a branch publish deploys nothing until its pin reaches `main`. The dev pair is for work *on* the machinery, and a dev store starts empty and can be wiped. The concrete motivation was the `hf`-marked integration tests, which wrote probe commits into the production publish repo every run (49 → 57 commits over one afternoon of running them). They now pick the `dev` profile themselves whenever one is defined, rather than the test runner exporting `MINI_PROFILE`, so a plain `pytest -m hf` is safe in any checkout that has the table, and unchanged in one that doesn't.

## Two lock files, one identity record

`publish.lock` says which revision the site serves; CI's build and its `Reports published` check read it and nothing else. Under a profile, pins go to a gitignored `.mini/publish.<profile>.lock` instead. So a dev publish gets no PR preview (the local `./go preview` is the thing under test in engineering work), and a `./go publish` run under `dev` on a science branch leaves the production pin unmoved, which the pre-push hook then reports as unpublished: the right signal for "you published to the wrong place". The alternative, lock entries that name their repo so a PR preview could serve a dev pin, is a schema change and a new CI rule bought for previews of engineering work; not worth it for this purpose.

## The backup is a separate trust domain

Every development environment holds tokens with write on the repo, the bucket, or the publish repo. A backup those tokens can reach is not a backup from the incident it is for. So the backup is a *pull*: a separate GitHub repo runs a nightly Actions job that fetches from the sources and writes into itself (with the job's own `GITHUB_TOKEN`, scoped to that repo) and into one HF dataset repo (with a token that has read on the sources and write on the backup, held only in that repo's Actions secrets). A dataset rather than a bucket, because the backup wants history.

Two rules keep it trustworthy. It never deletes: no `delete_patterns`, no squash, nothing removed on either side, so a mistake upstream cannot propagate. And it never runs code from the sources: the script and workflow are the backup repo's own copy of `templates/backup/`; a script pulled from the mirrored head would let write access on the source rewrite the backup job itself.

Three legs, shaped by what each source is. The code leg is plain git: a `mirror` branch fast-forwarded to the source's `main`, a `snap/<date>` tag whenever the tip moved, and a ruleset on the backup repo (no bypass list on a personal repo, so it binds the job's token too) that makes tags immutable and forbids force pushes. The store leg relies on the CAS being write-once-by-hash: whatever the backup lacks is the whole delta, `refs/` is refreshed each run, and earlier pointers live on in the backup's history. The publish leg replays the source's commits oldest-first, one backup commit each, with the source sha in the commit title and in a `pub/SOURCE_COMMIT` marker committed alongside the files. The head of `pub/` is the union of every replayed revision, and a pinned revision is recovered from the backup commit that replayed it. A marker that names a commit no longer in the source's history means the source was rewritten; the leg then stops and reports rather than replaying a rewritten past over the record of the real one.

Which account owns the backup repos is the one choice the template leaves open. Sibling repos under the project's account are outside the reach of a leaked *token*; a compromised *login* needs a second account. Past that lies object lock (S3 or B2 compliance mode), where even the account cannot delete before retention ends: a fourth leg for a project whose data would hurt to lose, and a second provider to run, so left out here.

Keeping a public repo's schedule alive: GitHub disables it after 60 days without repository activity, and the job's nightly commit of `state/last-run.json` is that activity. Whether a `GITHUB_TOKEN` commit counts is not stated in GitHub's docs; the keepalive actions people rely on for this work by the same mechanism, and a private repo is exempt regardless.

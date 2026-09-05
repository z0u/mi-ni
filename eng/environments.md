# Environments: production, dev, and the backup

*Part of the [engineering notes](./README.md).*

Three places a project's bytes live, and why each is where it is. The *how* is in the `storage-envs` and `backup` skills.

## The profile picks names; the token draws the boundary

A `[tool.mini.profiles.<name>]` table, selected by `MINI_PROFILE`, replaces the two storage keys (`store-bucket`, `publish-repo`) and inherits the rest.

*Replaces, never merges the pair.* A profile that names only a bucket has no publish repo; it does not fall back to the production one. If the missing key were inherited, the first half-written profile would send a dev publish into the production repo, and the failure would look like success. "Unset" is what a project without the key gets anyway (a local store, or single-bucket publishing), so the safe behavior is also the ordinary one.

*Inherits everything else.* The first cut replaced the whole table. Then `MINI_PROFILE=dev` dropped `app`, `env`, and `region`, and the CLI forgot its backend. Those keys describe the compute, which a sandbox shares with production. Only the two storage names are what a sandbox exists to change.

The real boundary is the credential. An engineering environment holds a token with write access on the dev pair only, so a session that forgets the profile fails on its first write. Some environments configure storage by variable instead: a Claude Code web environment sets `MINI_STORE_BUCKET` and `MINI_PUBLISH_REPO`, and has no `mini.local.toml`. Those need no profile at all. Point the two variables at the dev pair and put the dev token beside them.

There is no promotion step, by design. Science runs and their reports always use production, and the publish tier already stages them: a branch publish deploys nothing until its pin reaches `main`. The dev pair is for work *on* the machinery, and a dev store starts empty and can be wiped. What prompted this was the `hf`-marked integration tests, which wrote probe commits into the production publish repo on every run (49 → 57 commits over one afternoon). They now pick the `dev` profile themselves whenever one is defined, rather than relying on the test runner to export `MINI_PROFILE`. So a plain `pytest -m hf` is safe in any checkout that has the table, and unchanged in one that doesn't.

## Two lock files, one identity record

`publish.lock` says which revision the site serves. The CI build and the `Reports published` check read it and nothing else. Under a profile, pins go to a gitignored `.mini/publish.<profile>.lock` instead.

Two things follow. A dev publish gets no PR preview, which is fine, because in engineering work the thing under test is the local `./go preview`. And a `./go publish` run under `dev` on a science branch leaves the production pin unmoved, so the pre-push hook reports the report as unpublished. That is the right signal for "you published to the wrong place".

We could instead have lock entries name their repo, so a PR preview could serve a dev pin. That means a schema change and a new CI rule, just to preview engineering work, so we skipped it.

## The backup is a separate trust domain

Every development environment holds tokens that can write to the repo, the bucket, or the publish repo. If one of those tokens leaks, a backup it can reach is no protection. So the backup *pulls* instead. A separate GitHub repo runs a nightly Actions job that fetches from the sources and writes into three targets under the backup account: a mirror repo on GitHub, a dataset on the Hub, and a bucket on the Hub. The dataset is where history matters (the publish repo's commits, the store's mutable pointers); the bucket is where forgetting matters (the store's blobs, which the source itself is meant to shed).

Two rules keep it trustworthy. It never deletes anything the source still has, and nothing sooner than a retention window: no `delete_patterns`, no squash, no forced push, so a mistake upstream cannot propagate faster than a season. And it never runs code from the sources. The script and workflow are the backup repo's own copy of `templates/backup/`. A script pulled from the mirrored head would let anyone with write access on the source rewrite the backup job itself.

The sources and the backup belong to different accounts. A fine-grained token on either service reaches only the repos its own account can see, so no single token spans both. Reads use read-only tokens from the account that owns the sources, and no token at all for public sources. Writes use credentials from the backup account. For the Hub targets that is a token minted fresh each run from the target's trusted publisher, which trusts the backup repo, branch, and workflow file; each token lives an hour and reaches one target. For the mirror repo it is one stored fine-grained token, for a reason given below. The script keeps separate clients, and the source client never falls back to a write token.

There are three legs, each shaped by what its source is.

The code leg is git. A `mirror` branch in the mirror repo is fast-forwarded to `main` on the source, and a `snap/<date>` tag is written whenever the tip moved. A ruleset on the mirror repo makes tags immutable and forbids force pushes, with an empty bypass list so it binds the job's token too.

The mirror is a second repo, and the push uses a personal token, because of one rule and its consequence. The job's own `GITHUB_TOKEN` may not write under `.github/workflows/`, and no permission grants it, so a source that has workflows cannot be mirrored into the backup repo at all; stripping the files would rewrite every sha and with it the tip the snapshot records. A fine-grained token with workflows write can push them, but a push made with a personal token triggers workflows, where a push made with the job's token does not, and the commits being pushed carry the source's own. So the mirror lives in a repo with Actions disabled, where nothing pushed can run, the token is scoped to that one repo (which holds no secrets, and whose rulesets leave the token able only to append), and the leg reads the Actions setting through the API before each push and refuses while it is on. We first considered a `git bundle` into the dataset instead, which needs no GitHub secret and sidesteps the rule entirely; we chose the repo because a browsable commit log is worth one scoped, expiring secret.

The store leg relies on the store being write-once by hash: a given file always lands at the same address, and is never rewritten.[^cas] So whatever the backup lacks is the whole delta, and the Hub copies bucket files into a bucket server-side by content hash, so the delta costs a listing and a few API calls rather than bytes through the runner. The first design copied the store into the dataset; that moved every byte twice, and it kept every blob for ever, when the source is meant to shrink under `mini gc --store`. The backup bucket instead follows the source with a delay: a file the source no longer has is kept for a retention window (90 days, against gc's 14-day grace) and then deleted, and the date each file was first missed lives in `state/store-missing.json` in the backup repo. The clock is one the source can start but not hurry, so a mass deletion upstream gives the whole window to notice, and a lost manifest only delays expiry. `refs/` are the exception to write-once: they are overwritten in place, and a bucket forgets the old value at once, so a copy of them also goes into the dataset under `store/refs/`, where every version stays.

The publish leg replays the commits of the source oldest-first, one backup commit each. The source sha goes in the commit title, and in a `pub/SOURCE_COMMIT` marker committed alongside the files. The head of `pub/` is the union of every replayed revision, and a pinned revision is recovered from the backup commit that replayed it. If a marker names a commit that is no longer in the history of the source, the source was rewritten. The leg then stops and reports, rather than replaying a rewritten past over the record of the real one.

All three legs live in one script. That way the git behaviors (a refused fast-forward, a moved tag, a same-day rerun) are unit-tested against local repos rather than trusted to a shell step, and the retention clock is unit-tested as a pure function of two listings and a date.

The template leaves the owning account open; for mi-ni we chose a second account. Sibling repos under the same account as the project are already out of reach of a leaked *token*, but a compromised *login* reaches them, and only a second account does not. A plus-addressed email counts as a distinct account on both services and still lands in the same inbox, and GitHub's terms allow one machine account beside a personal one. Free organizations don't give the same property: on GitHub an organization is owned by the personal account, and on Hugging Face the token policies that would fence one off are a paid feature. Beyond that lies object lock (S3 or B2 compliance mode), where even the account cannot delete before retention ends. That would be a fourth leg for a project whose data would hurt to lose. We left it out because it means running a second provider.

A public repo also needs its schedule kept alive. GitHub disables the schedule after 60 days without repository activity, and the nightly commit of `state/last-run.json` counts as activity. The docs don't say whether a `GITHUB_TOKEN` commit counts, but the keepalive actions people rely on work the same way, and a private repo is exempt regardless.

[^cas]: A content-addressed store names each file by the hash of its contents, so the name changes whenever the contents do.

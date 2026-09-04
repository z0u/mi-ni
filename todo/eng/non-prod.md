---
status: open
tags: [devops, security, publishing, storage]
opened: 2026-09-04
bundle: env-hardening
---

# Create a non-prod environment

We currently have a single bucket and dataset repo that we push to. Create a second one to use for development. This should include a mechanism that can be used by forks and users of the template repo.

## Proposed design (for review, not yet built)

**What it is for: an engineering sandbox, never a science environment.** Science runs, and the reports they feed, always use the production pair; the publish tier already stages those (a branch publish deploys nothing until its pin reaches `main`, and the PR preview serves the branch's pins). The dev pair exists so that work *on* the storage and publishing machinery can run against real Hugging Face repos without touching production: the `hf`-marked integration tests (which today write probes under `refs/_test/` in the production bucket), a `sync_export` or `gc --store` change mid-development, an agent building a store feature, a template user trying the pipeline before they have data. Because nothing in dev is ever meant to reach production, there is no promotion step and no cross-environment read; a dev store starts empty and can be wiped.

**Config: a named profile, opt-in.**

```toml
[tool.mini]
store-bucket = "z0u/mi-ni-store"          # production, as now
publish-repo = "z0u/mi-ni-pub"

[tool.mini.profiles.dev]
store-bucket = "z0u/mi-ni-store-dev"
publish-repo = "z0u/mi-ni-pub-dev"
```

`MINI_PROFILE=dev` selects the table; unset means the base keys, so nothing changes for a project without profiles. `MINI_STORE_BUCKET` and `MINI_PUBLISH_REPO` still override everything, `mini.local.toml` overlays profile tables the way it overlays the base keys, and the Modal secret already forwards the resolved names, so a worker follows its driver's profile. `./go auth --check` and `./go publish` print the active profile and target repos. A fork or template user fills in their own pairs, or leaves them unset and stays on the local store, as now. Both dev repos can be private; only the site build needs anonymous reads, and it never sees dev.

**The credential is the boundary.** The profile picks names; the token decides what can be written. An environment set aside for engineering work (a devcontainer, a Claude Code web environment, the test job in CI) carries a token with write on the dev pair only, so a session that forgets `MINI_PROFILE=dev` fails on its first write instead of succeeding quietly. Science environments keep their production tokens; the backups item covers what happens if one leaks.

**The lock file: dev pins never enter `publish.lock`.** `publish.lock` is the production identity record, the file that says which revision the site serves. Under a non-base profile, `./go publish` writes its pins to `.mini/publish.dev.lock` instead (gitignored by the existing `.mini/` rule), and `./go preview` and a local `./go site` read the active profile's lock. CI's build and its `Reports published` check are untouched: they only ever see production pins. Two consequences worth stating. A dev publish gets no PR preview in CI, which is fine for engineering work, where the local preview is the thing under test. And running `./go publish` under `dev` on a science branch leaves that report's production pin unmoved, so the pre-push hook flags it as unpublished, which is the right signal for "you published to the wrong place". The alternative, lock entries that name their repo so the build can serve a dev pin in a PR preview, buys previews of engineering work at the cost of a schema change and a new CI rule, and I would not pay it for this purpose.

**What stays shared, and one thing to watch.** Modal control-plane state (`Dict`s, per-experiment Volumes, the HF cache Volume) is named per experiment, so a dev run of an experiment named `gpt-sweep` would share Modal state with a production run of the same name. Engineering runs usually use throwaway names, so I would leave this alone at first and prefix Modal names with the profile only if it bites. `mini gc --store` sweeps whichever bucket the active profile names, so dev junk is reclaimed the usual way, or the dev bucket is simply emptied.

**Tests.** `scripts/test.sh` (or the `hf` marker's conftest) sets `MINI_PROFILE=dev` when the profile exists, so the integration suite stops writing to production; the `bucket_publish` skip logic in `tests/mini/test_hf_store.py` then reads the dev pair's visibility rather than production's.

**Setup as a skill.** The profile mechanism is library usage, so it goes in the `mi-ni` skill's `references/storage.md` beside `store-bucket`/`publish-repo`; the setup steps (create the two dev repos, add the table, mint a dev-only token for the engineering environments, point the test runner at the profile) go in a short `.agents/skills/storage-envs/SKILL.md` that the backup skill in `backups.md` can cross-link, since the two are set up together.

**Effort.** Config resolution and the two status prints (small), the lock-path switch in `export_reports.py`/`build_site.py` (small), the test wiring (small), the storage skill reference and `docs/README.md`, and two new HF repos. A day's work, most of it in `store.py`. Depends on `publish-tier-hardening.md` being finished, so that the production bucket a dev token cannot write is also one the world cannot read.

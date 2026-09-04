---
status: open
tags: [devops, security, publishing, storage]
opened: 2026-09-04
bundle: env-hardening
---

# Create a non-prod environment

We currently have a single bucket and dataset repo that we push to. Create a second one to use for development. This should include a mechanism that can be used by forks and users of the template repo.

## Proposed design (for review, not yet built)

**What it is for.** Two things, and it helps to name them because the existing design already covers a third. The publish tier is already staged: a publish from a branch deploys nothing until its pin in `publish.lock` reaches `main`, and PR previews serve the branch's pins (`eng/publishing.md`). What a second environment adds is (1) a smaller blast radius, since only one environment needs a token that can write the production stores, and (2) a place for branch and agent work to write refs and blobs without touching what production reports and citations resolve from. Today every agent session holds a production write token; the backups item lists that as the risk it exists to absorb, and this item removes the capability instead.

**Config: profiles, selected by an environment variable, with the base table as the default.**

```toml
[tool.mini]
store-bucket = "z0u/mi-ni-store-dev"     # what a fresh clone or agent session talks to
publish-repo = "z0u/mi-ni-pub-dev"

[tool.mini.profiles.prod]
store-bucket = "z0u/mi-ni-store"
publish-repo = "z0u/mi-ni-pub"
```

`MINI_PROFILE=prod` selects the table; unset means the base keys, so a project with one store keeps working unchanged. `MINI_STORE_BUCKET` and `MINI_PUBLISH_REPO` still override everything, `mini.local.toml` overlays profiles like it overlays the base keys, and the Modal secret already forwards the resolved names, so workers follow the driver's profile. The repo decides which environment is the default by which pair it writes in the base table; putting dev there is the fail-closed choice, and `./go auth --check` prints the active profile so it is never a surprise. A fork or template user fills in their own pairs, or leaves them unset and stays on the local store, as now.

**The environment is really the credential.** The profile picks names; the token decides what can be written. Development environments (devcontainer, Claude Code web, CI) get a fine-grained HF token with write on the dev pair and read on the prod pair. Only Sandy's machine holds a token with write on the prod pair. A misconfigured dev session that selects `prod` then fails on the first write rather than succeeding quietly.

**Promotion, so that dev work becomes production without re-running it.** `./go promote <report>...`, run from the one environment with the prod token, does three things per report from the bundle already published to the dev repo: read its `_assets/provenance.json` for the refs the render resolved, copy those refs and their blobs into the prod bucket (`put` is idempotent by hash, and Xet dedup makes the transfer metadata-sized), and upload the bundle bytes unchanged into the prod publish repo. It rewrites the report's `publish.lock` entry to the prod sha and leaves the commit for the branch. No notebook runs, so the preview that was reviewed is byte-for-byte what production serves, and prod's refs and CAS hold the data behind every published figure, which keeps provenance and citations resolvable from prod alone.

**The lock names its repo.** A pin recorded by a dev publish is a sha in the dev repo, so `publish.lock` entries gain the repo they point into (`{"rev": sha, "repo": "z0u/mi-ni-pub-dev"}`, with a bare string still read as the base publish repo). The site build fetches each bundle from the repo its entry names; both repos are public, so CI's read-only token covers them. The existing `Reports published` check gains one rule: on `main`, every entry must name the prod repo. That makes promotion a merge gate that only a human's machine can pass, which is the security property restated as workflow: agent runs the experiment and publishes to dev, the PR preview shows it, Sandy reviews and promotes, the PR merges.

**Cross-environment reads.** A dev session that wants a production result (to build on a sweep, say) has no path to it in the first cut; the profiles are separate stores. Two options for later, in rising order of ambition: seed the dev bucket from the vault's copy (the backups item's restore drill, made useful), or a read-through fallback in `HFStore` where a `get` that misses in the profile's store tries the prod pair read-only. The first needs no code. I would hold the second until a recompute actually hurts.

**What stays shared.** Modal control-plane state (`Dict`s, per-experiment Volumes, the HF cache Volume) is named per experiment and per checkout, so it is already dev-shaped and needs no profile suffix. `mini gc --store` sweeps whichever bucket the active profile names, so dev junk is reclaimed the usual way.

**Effort.** Config resolution and `auth --check` (small), lock schema and build-side repo lookup (small), the `Reports published` rule (small), `promote` (the substantial piece, roughly 150 lines, most of it reuse of `publish`/`sync_export`), plus the storage skill reference and `docs/README.md`. Two new HF repos to create. Depends on `publish-tier-hardening.md` being finished first (private bucket), since a private prod CAS is what the dev/prod token split protects.

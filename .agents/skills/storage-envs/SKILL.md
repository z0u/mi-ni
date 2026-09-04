---
name: storage-envs
description: Set up a non-production (dev) pair of Hugging Face repos for a mi-ni project — a store bucket and a publish repo beside the production ones — and point the engineering environments and the integration tests at it. Use when the `hf` tests should stop writing to production, when narrowing an environment's token, or when a template user wants a sandbox.
---

# A dev pair beside production

The mechanism is a `[tool.mini.profiles.<name>]` table selected by `MINI_PROFILE`, described in the `mi-ni` skill's [storage reference](../mi-ni/references/storage.md#profiles-a-dev-pair-beside-production); this skill is the setup. The pair is an engineering sandbox. Science runs, and the reports they feed, always use production (the publish tier already stages those through `publish.lock`), so nothing in dev is ever promoted, and a dev pair can be wiped at any time.

## Steps

1. Create the two repos on Hugging Face, in the production namespace and named after it: a bucket and a dataset repo, `-dev` suffixed. Both can be private; only the site build needs anonymous reads, and it never sees dev. Creating a repo is a namespace-level permission that a per-repo token lacks, so this is usually the human's step:

   ```bash
   uv run hf repos create <ns>/<pub>-dev --type dataset --private
   uv run python -c "from huggingface_hub import HfApi; HfApi().create_bucket('<ns>/<store>-dev', private=True)"
   ```

2. Add the profile table. Where the production pair is committed in `pyproject.toml`, add `[tool.mini.profiles.dev]` there too, so the profile travels with the repo. Where the pair lives in the gitignored `mini.local.toml` (this template's own checkout), the profile table goes in the same file.

3. Mint a dev-only token (human): a fine-grained Hugging Face token with read and write on the two dev repos and nothing else. It goes into the environments set aside for engineering work: a devcontainer, a Claude Code web environment, the CI test job. Science environments keep their production tokens.

4. Point those environments at dev. A file-configured checkout sets `MINI_PROFILE=dev` in its environment (the devcontainer's env, a shell profile). An environment that configures storage by variable rather than by file, as a Claude Code web environment does with `MINI_STORE_BUCKET` and `MINI_PUBLISH_REPO`, has no table to select from: set those two to the dev names instead, beside the dev token. Either way the token is what makes forgetting safe: a session on a dev token that reaches for production fails on its first write.

5. Check. `./go auth --check` shows `profile dev` and the dev bucket, and `uv run pytest -m hf` runs against the pair. The integration tests choose the `dev` profile themselves whenever one is defined, so they stop writing to production the moment the table exists; without one they use the active profile, or production, as before.

## What stays shared

Modal control-plane state (`Dict`s, per-experiment Volumes, the HF cache Volume) is named per experiment, so a dev run of an experiment named like a production one shares Modal state with it. Engineering runs use throwaway names; prefix Modal names with the profile only if that bites. `mini gc --store` sweeps whichever bucket the active profile names, so dev junk is reclaimed the usual way.

Seeding a dev pair from the project's backup is a restore drill with a use: the `backup` skill.

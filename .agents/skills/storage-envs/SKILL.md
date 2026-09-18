---
name: storage-envs
description: Set up a dev pair of Hugging Face repos (store bucket and publish repo) beside a mi-ni project's production pair, and point the engineering environments and the `hf` tests at it.
---

# A dev pair beside production

A `[tool.mini.profiles.<name>]` table names the pair, and `MINI_PROFILE` selects it. The [storage reference](../mi-ni/references/storage.md#profiles-a-dev-pair-beside-production) in the `mi-ni` skill describes how that works; this skill covers the setup.

Treat the dev pair as an engineering sandbox. Science runs and the reports they feed always use production, which the publish tier already stages through `publish.lock`. So nothing in dev is ever promoted, and you can wipe a dev pair at any time.

## Steps

1. Create two repos on Hugging Face, a bucket and a dataset repo. Put them in the production namespace, and give each the production name plus a `-dev` suffix. Both can be private: only the site build needs anonymous reads, and it never sees dev. Creating a repo needs a namespace-level permission that a per-repo token lacks, so this is usually the human's step:

   ```bash
   uv run hf repos create <ns>/<pub>-dev --type dataset --private
   uv run python -c "from huggingface_hub import HfApi; HfApi().create_bucket('<ns>/<store>-dev', private=True)"
   ```

2. Add the profile table, in whichever file holds the production pair. If that is `pyproject.toml`, add `[tool.mini.profiles.dev]` there too, so the profile travels with the repo. If the pair lives in the gitignored `mini.local.toml`, as it does in this template's own checkout, put the profile table in that file. Name the Modal Environment in it too (`modal-environment = "dev"`), so a dev run's memo records and Volumes live apart from production's; without it, a dev run of an experiment that shares a name with a production one finds production's memo records and skips the work (`eng/environments.md`).

3. Create the Modal Environment. Any session with a Modal token can do this: `uv run modal environment create dev`. Nothing else is created up front; the `Dict`s and Volumes appear on the first dev run.

4. Mint a dev-only token (human): a fine-grained Hugging Face token with read and write on the two dev repos and nothing else. It goes into the environments set aside for engineering work, such as a devcontainer, a Claude Code web environment, or the CI test job. Science environments keep their production tokens.

5. Point those environments at dev. A checkout configured by file sets `MINI_PROFILE=dev` in its environment, such as the devcontainer env or a shell profile. Some environments configure storage by variable rather than by file; a Claude Code web environment uses `MINI_STORE_BUCKET`, `MINI_PUBLISH_REPO` and `MODAL_ENVIRONMENT`. Those have no table to select from, so set the two variables to the dev names, beside the dev token. Either way, the token is what makes forgetting safe: a session on a dev token that reaches for production fails on its first write.

6. Check. `./go auth --check` should show `profile dev`, the dev pair, and the Modal Environment, and `uv run pytest -m hf` should run against the pair. The integration tests pick the `dev` profile themselves whenever one is defined, so they stop writing to production as soon as the table exists. Without a dev profile they use the active profile, or production, as before.

## What stays shared

Modal control-plane state (`Dict`s, per-experiment Volumes, the HF cache Volume) is named per experiment. So a dev run of an experiment that shares a name with a production one shares Modal state with it too. Engineering runs use throwaway names; prefix Modal names with the profile only if the sharing causes trouble. `mini gc --store` sweeps whichever bucket the active profile names.

You can seed a dev pair from the project backup, which doubles as a restore drill. See the `backup` skill.

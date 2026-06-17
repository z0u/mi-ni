#!/usr/bin/env bash
#
# SessionStart hook for Claude Code on the web.
#
# The web base image ships older tooling than this project assumes. This hook
# brings it in line — the web-runtime analogue of .devcontainer/post-create.sh.
# Safe to run repeatedly; the container state is cached after it completes, so
# subsequent sessions skip the slow paths.
#
set -euo pipefail

# Web-only. Local checkouts and the dev container manage their own tooling
# (the devcontainer installs uv via a feature), so don't interfere there.
if [[ "${CLAUDE_CODE_REMOTE:-}" != 'true' ]]; then
    exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

log() { echo "session-start: $*" >&2; }

# 1. Ensure uv is new enough to parse pyproject.toml.
#    The image ships uv 0.8.x, which can't parse the relative
#    `exclude-newer = "N days"` cooldown and warns on every `uv run`. uv added
#    relative durations later. We upgrade from PyPI rather than via
#    `uv self update`, because that checks GitHub releases and the network
#    policy here blocks the GitHub API (403).
min_uv='0.11'
have_uv="$(uv --version 2>/dev/null | awk '{print $2}' || echo '0')"
if [[ "$(printf '%s\n%s\n' "$min_uv" "$have_uv" | sort -V | head -n1)" != "$min_uv" ]]; then
    log "upgrading uv ${have_uv} -> latest (from PyPI)"
    uv tool install uv --force >/dev/null 2>&1 || log "uv upgrade failed; continuing with ${have_uv}"
fi

# 2. Sync the project venv so linters, type-checker, tests, and notebooks work.
#    Mirrors `./go install` (minus npm/git-hooks, which the agent doesn't need).
#    --no-group cuda: locally we run CPU-only; the CUDA plugin is for Modal.
log "syncing venv (uv $(uv --version 2>/dev/null | awk '{print $2}'))"
uv sync --all-groups --no-group cuda >/dev/null 2>&1 || log 'uv sync failed; venv may be incomplete'

log 'ready'
exit 0

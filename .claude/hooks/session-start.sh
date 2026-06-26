#!/usr/bin/env bash
#
# SessionStart hook for Claude Code on the web.
#
# The web base image ships older/leaner tooling than this project assumes. This
# hook brings it in line — the web-runtime analogue of .devcontainer/post-create.sh.
# It does three things: point the shell at the project's Python, sync the venv,
# and install the agent-friendly CLI tools the dev container ships as features.
#
# The venv sync runs asynchronously (see the {"async": true} line below): the
# session starts immediately while setup proceeds in the background. The PATH
# export is written *before* that handoff so it always lands for this session.
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
project_dir="$PWD"

log() { echo "session-start: $*" >&2; }

# 0. Put the project venv first on PATH so bare `python` resolves to the
#    project's 3.14 interpreter instead of the image's system Python 3.11 (which
#    chokes on 3.14-only syntax), and so `ruff`, `ty`, `pytest`, `marimo`,
#    `mini`, and `modal` are callable without the `uv run`/`./go` prefix.
#    CLAUDE_ENV_FILE is sourced into the session's shells by Claude Code; we
#    write it synchronously, before the async handoff below, so it applies even
#    though the heavy setup is still running. $PATH stays unexpanded (single
#    quotes via escaping) so it composes with each shell's PATH; the guard keeps
#    repeated SessionStart events (resume/clear/compact) from stacking entries.
if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
    if ! grep -qs "$project_dir/.venv/bin" "$CLAUDE_ENV_FILE"; then
        echo "export PATH=\"$project_dir/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
        log "added .venv/bin to PATH"
    fi
fi

# Everything below is slow; run it in the background so the session starts now.
# `uv run` syncs on demand anyway, so the worst case is the first command
# re-doing work this hook is also doing.
echo '{"async": true, "asyncTimeout": 600000}'

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

# 3. Install the agent-friendly CLI tools the dev container ships as features
#    but the web image lacks: fd, fzf, bat. (rg is already present; gh is
#    omitted because this network policy blocks the GitHub API — use the GitHub
#    MCP tools instead.) Debian packages the binaries as fdfind/batcat to avoid
#    name clashes, so symlink the names CLAUDE.md actually uses. The whole hook
#    is web-only (guarded above), but gate on apt-get too so this degrades
#    cleanly should the cloud base image ever not be Debian-based.
if command -v apt-get >/dev/null 2>&1 \
    && ! { command -v fd && command -v bat && command -v fzf; } >/dev/null 2>&1; then
    log 'installing fd, fzf, bat'
    if DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        fd-find fzf bat >/dev/null 2>&1; then
        [[ -e /usr/bin/fdfind ]] && ln -sf /usr/bin/fdfind /usr/local/bin/fd
        [[ -e /usr/bin/batcat ]] && ln -sf /usr/bin/batcat /usr/local/bin/bat
    else
        log 'apt install of fd/fzf/bat failed; continuing'
    fi
fi

log 'ready'
exit 0

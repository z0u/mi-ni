#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

URL_MODE=open
CHECK=
for arg in "$@"; do
    case "$arg" in
        --qr) URL_MODE=qr ;;
        --check|--status|-c) CHECK=1 ;;
    esac
done

# `--check`: just probe the tokens (in parallel) and print their statuses —
# no interactive setup, no secrets echoed. Handy for confirming Modal/HF are
# already authenticated without poking the raw tools.
if [[ -n "$CHECK" ]]; then
    exec uv run "$SCRIPT_DIR/auth_check.py"
fi

intercept() {
    uv run "$SCRIPT_DIR/intercept_urls.py" "--$URL_MODE" "$@"
}

show_url() {
    uv run "$SCRIPT_DIR/intercept_urls.py" "--$URL_MODE" --url "$1"
}

# Modal
if ! uv run modal token info &>/dev/null; then
    intercept modal setup
fi
echo "✅ Modal authenticated"

# Hugging Face — fine-grained token for the artifact store bucket (see the mi-ni
# storage reference). Reads/writes the project's content-addressed store and
# serves published figures. `hf` caches the token, and both the store and the
# Modal worker pick it up from there.
if ! uv run hf auth whoami &>/dev/null; then
    show_url "https://huggingface.co/settings/tokens/new?tokenType=fineGrained&tokenName=mi-ni+store"
    echo "Create a fine-grained token with read & write access to your store bucket"
    echo "(under 'Repositories', select the bucket), then paste it below."
    read -rsp 'Token: ' hf_token
    echo
    uv run hf auth login --token "$hf_token"
fi
echo "✅ Hugging Face authenticated"

# Claude Code
if ! claude auth status &>/dev/null; then
    claude auth login
fi
echo "✅ Claude Code authenticated"

# GitHub — fine-grained PAT scoped to just this repo, stored in gh's config.
if ! gh auth status &>/dev/null; then
    owner="$(git -C "$SCRIPT_DIR/.." remote get-url origin 2>/dev/null \
        | sed -E 's#.*[:/]([^/]+)/[^/]+(\.git)?$#\1#')"
    show_url "https://github.com/settings/personal-access-tokens/new?name=mi-ni+agent&description=mi-ni+agent&target_name=${owner}&expires_in=30&contents=write&issues=write&pull_requests=write&actions=read"
    echo "Under 'Repository access' pick 'Only select repositories' → this repo, confirm the owner is '${owner}', then generate and paste the token."
    read -rsp 'Token: ' gh_token
    echo
    echo "$gh_token" | gh auth login --with-token
fi
echo "✅ GitHub authenticated"

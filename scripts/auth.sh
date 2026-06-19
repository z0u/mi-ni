#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

URL_MODE=open
for arg in "$@"; do
    [[ "$arg" == "--qr" ]] && URL_MODE=qr
done

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

# WandB
if uv run wandb status 2>/dev/null | grep -q '"api_key": null'; then
    intercept wandb login
fi
echo "✅ WandB authenticated"

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

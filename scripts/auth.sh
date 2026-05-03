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

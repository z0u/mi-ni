#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

( set -x; uv run "$SCRIPT_DIR/auth.py" "$@" )

echo "✅ Authenticated"

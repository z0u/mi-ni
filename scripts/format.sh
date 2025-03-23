#!/usr/bin/env bash

set -euo pipefail

( set -x; uv run ruff format "$@" )

echo "✅ Formatted"

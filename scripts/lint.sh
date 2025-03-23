#!/usr/bin/env bash

set -euo pipefail

( set -x; uv run ruff check "$@" )

echo "✅ Lint check passed"

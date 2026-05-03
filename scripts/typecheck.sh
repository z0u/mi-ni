#!/usr/bin/env bash

set -euo pipefail

( set -x; uv run ty check "$@" )

echo "✅ Type check passed"

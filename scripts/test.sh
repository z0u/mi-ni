#!/usr/bin/env bash

set -euo pipefail

( set -x; uv run pytest "$@" )

echo "✅ Tests passed"

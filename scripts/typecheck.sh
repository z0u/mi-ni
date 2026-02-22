#!/usr/bin/env bash

set -euo pipefail

( set -x; uv run pyrefly check "$@" )

echo "✅ Type check passed"

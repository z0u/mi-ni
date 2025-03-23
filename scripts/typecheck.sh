#!/usr/bin/env bash

set -euo pipefail

( set -x; uv run pyright "$@" )

echo "✅ Type check passed"

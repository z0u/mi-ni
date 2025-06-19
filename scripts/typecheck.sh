#!/usr/bin/env bash

set -euo pipefail

( set -x; uv run basedpyright "$@" )

echo "✅ Type check passed"

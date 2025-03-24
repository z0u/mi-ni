#!/usr/bin/env bash

set -euo pipefail

( set -x; uv run modal setup "$@" )

echo "✅ Authenticated"

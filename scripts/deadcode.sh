#!/usr/bin/env bash

set -euo pipefail

(
    set -x
    mkdir -p .vulture-cache
    rm -r .vulture-cache/* || true
    uv run python scripts/ipynb_to_py.py *.ipynb .vulture-cache/
    uv run vulture "$@"
)

echo "✅ Dead code check passed"

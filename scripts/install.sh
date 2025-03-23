#!/usr/bin/env bash

set -euo pipefail

# Check if we're installing for CPU-only or with CUDA support
if [[ "${1:-}" == "cpu" ]]; then
    echo "Installing with CPU-only PyTorch packages..." >&2
    ( set -x; uv sync --all-groups --extra torch-cpu < /dev/null )
else
    echo "Installing with default PyTorch (based on current platform). See https://pytorch.org/get-started/locally/" >&2
    ( set -x; uv sync --all-groups --extra torch < /dev/null )
fi

echo "✅ Installation complete"

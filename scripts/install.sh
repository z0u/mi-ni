#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Process options

show_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --help                show this help message"
}

# Handle arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      show_usage
      exit 0
      ;;
    *)
      echo "Error: Unknown option '$1'" >&2
      show_usage
      exit 1
      ;;
  esac
  shift
done

( set -x; uv sync --all-groups < /dev/null )


echo "✅ Installation complete"

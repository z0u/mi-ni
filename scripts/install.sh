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

# The cuda group is for remote (Modal) execution; locally we use CPU jax.
( set -x; uv sync --all-groups --no-group cuda < /dev/null )

( set -x; npm install )

# Install versioned git hooks
HOOKS_SRC="$SCRIPT_DIR/hooks"
HOOKS_DST="$SCRIPT_DIR/../.git/hooks"
if [[ -d "$HOOKS_DST" ]]; then
    for hook in "$HOOKS_SRC"/*; do
        name="$(basename "$hook")"
        ln -sf "../../scripts/hooks/$name" "$HOOKS_DST/$name"
        echo "Installed git hook: $name"
    done
fi

echo "✅ Installation complete"

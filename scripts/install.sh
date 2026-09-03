#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Process options

# Hold the lockfiles fixed, or let the resolver rewrite them? Plain `uv sync` and
# `npm install` re-resolve and rewrite their lockfile whenever it disagrees with the
# manifest, silently — so a dependency edit pushed without a re-lock would go green in
# CI, having tested a resolution nobody reviewed, and merge a pyproject.toml the
# committed lock no longer describes. `exclude-newer` is what sharpens that: it counts
# in relative days, so a CI re-lock weeks after the edit draws from a different eligible
# set than the developer had, which is the reproducibility claim the cooldown exists to
# make, arriving by the back door. So CI holds them fixed, and a stale lock is a red
# build naming its own remedy (`uv lock` / `npm install`). Locally the default is off:
# a half-finished dependency edit shouldn't be blocked from installing.
LOCKED="${CI:+1}"

show_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --locked, --no-locked fail on (or allow) a lockfile the manifest has outgrown"
  echo "                        [default: --locked under \$CI, --no-locked otherwise]"
  echo "  --help                show this help message"
}

# Handle arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --locked)
      LOCKED=1
      ;;
    --no-locked)
      LOCKED=
      ;;
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
( set -x; uv sync --all-groups --no-group cuda ${LOCKED:+--locked} < /dev/null )

# npm's spelling of the same distinction: `ci` installs the lockfile as written and
# errors if package.json has outgrown it, where `install` reconciles the two in place.
if [[ -n "$LOCKED" ]]; then
  ( set -x; npm ci )
else
  ( set -x; npm install )
fi

# Replay conflict resolutions: long-lived branches re-merge main and hit the same ones.
( set -x; git -C "$SCRIPT_DIR/.." config rerere.enabled true )

# Install versioned git hooks
HOOKS_SRC="$SCRIPT_DIR/hooks"
HOOKS_DST="$SCRIPT_DIR/../.git/hooks"
if [[ -d "$HOOKS_DST" ]]; then
    for hook in "$HOOKS_SRC"/*; do
        name="$(basename "$hook")"
        ln -sf "../../scripts/hooks/$name" "$HOOKS_DST/$name"
        echo "Installed git hook: $name"
    done

    # Skip mechanical reformats in `git blame`. GitHub honours this file by
    # default, but a local clone needs the config set (it can't be committed).
    git config --local blame.ignoreRevsFile .git-blame-ignore-revs
    echo "Configured blame.ignoreRevsFile"
fi

echo "✅ Installation complete"

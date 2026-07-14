#!/usr/bin/env bash

set -euo pipefail

# Paths and filters come from [tool.vulture] in pyproject.toml.
if ! (
    set -x
    uv run vulture "$@"
); then
    cat >&2 <<-'EOF'
		❌ Dead code found! See the report above. To fix, you can:
		 1. Remove the unused code,
		 2. Re-run with "--make-whitelist >> .vulture-allowlist.py" to ignore all, or
		 3. Add "# noqa" comments for the false positives.
		EOF
    exit 1
fi

echo "✅ Dead code check passed"

#!/usr/bin/env bash
#
# PreToolUse hook: run CI's fast checks before a `git push` so failures surface
# here instead of after a CI round-trip. Mirrors the lint/format/type steps of
# .github/workflows/lint-check.yml.
#
# Only lint + format + ty run here — they're seconds, and they're the usual CI
# tripwire (e.g. ty over tests/). Tests are intentionally left to CI: this hook
# is synchronous, so running the suite would stall every push and scale badly on
# a large repo. When a session is watching the PR, CI test failures get picked
# up there instead.
#
# Soft by design — it never wedges a productive push:
#   - bypass entirely with `git push --no-verify`;
#   - steps aside (allows) if the venv isn't ready yet or jq/git aren't present.
#
set -uo pipefail  # not -e: exit codes are managed explicitly below

payload="$(cat)"

# Read the command out of the tool payload. If jq is missing, fail open.
command -v jq >/dev/null 2>&1 || exit 0
cmd="$(printf '%s' "$payload" | jq -r '.tool_input.command // ""')"

# The PreToolUse matcher only scopes us to the Bash tool (it matches tool names,
# not the command), so the `git push` filtering has to happen here. Let
# everything else through untouched.
case "$cmd" in
    *'git push'*) ;;
    *) exit 0 ;;
esac

# Explicit bypass, mirroring git's own --no-verify escape hatch.
case "$cmd" in
    *'--no-verify'*) exit 0 ;;
esac

project="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$project" 2>/dev/null || exit 0

# Step aside if the toolchain isn't ready (e.g. the SessionStart sync is still
# running). Better to let the push through than to block on a half-built env.
[[ -x .venv/bin/ty && -x .venv/bin/ruff ]] || exit 0

# Fast checks: block the push on failure, feeding the errors back so they can be
# fixed in the same flow. Tests stay in CI (a watching session catches those).
if ! fast_out="$(./go check --lint --format --typecheck 2>&1)"; then
    {
        echo 'Pre-push checks failed (lint/format/ty) — CI gates on these too.'
        echo 'Fix them, then push again — or run `git push --no-verify` to skip.'
        echo 'Auto-fix lint/format: `./go check --fix`'
        echo
        echo "$fast_out"
    } >&2
    exit 2
fi

exit 0

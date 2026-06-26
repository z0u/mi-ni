#!/usr/bin/env bash
#
# PreToolUse hook: run CI's checks before a `git push` so failures surface here
# instead of after a CI round-trip. Mirrors .github/workflows/lint-check.yml.
#
#   - lint + format + ty (fast) BLOCK the push on failure: the push is stopped
#     and the failures are fed back so they can be fixed in the same flow.
#   - tests run too, but only WARN: the push proceeds and the failure is
#     surfaced next to the tool result (additionalContext), to be fixed before
#     merge rather than gating every push on a ~1 min suite.
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

# Only gate `git push`; let everything else through untouched.
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

# --- Fast checks: block on failure -----------------------------------------
# These are seconds, and they're the usual CI tripwire (e.g. ty over tests/).
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

# --- Tests: warn only ------------------------------------------------------
# The fast checks passed, so the push is going through regardless; run the suite
# and, if it fails, attach a warning next to the push result instead of blocking.
if ! test_out="$(./go check --test 2>&1)"; then
    printf '%s\n\n%s\n' \
        '⚠️ Pre-push tests FAILED — the push was allowed, but CI will gate on this. Fix before merging.' \
        "$test_out" \
    | jq -Rs '{hookSpecificOutput: {hookEventName: "PreToolUse", additionalContext: .}}'
fi

exit 0

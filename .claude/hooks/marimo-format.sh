#!/usr/bin/env bash
#
# PostToolUse hook: keep Marimo notebooks in the shape Marimo itself writes them.
#
# Marimo copies a variable's annotation from the cell that defines it onto the
# parameter list of every cell that reads it. We don't do that when we edit the
# file by hand — and we sometimes strip the annotations back off — so the file
# drifts from what Marimo serializes. The next `marimo edit` session writes them
# back, and they land in the human's working tree as churn they didn't author.
#
# `marimo check --fix` applies that same rewrite, so running it after every edit
# keeps the file matching what the kernel would save. It's authoritative over
# cell signatures: it fills in missing annotations, corrects wrong ones, and
# removes ones with no annotated definition behind them. To change a parameter's
# type, annotate the definition instead.
#
# This is the copy that runs everywhere Claude Code runs, including the web,
# where .git/hooks are absent (scripts/install.sh installs them, and the web
# SessionStart hook deliberately skips that step). The lint-staged entry in
# package.json covers the same ground for commits made by hand.
#
# Soft by design: it steps aside if jq or the venv aren't ready rather than
# failing an edit over formatting.
#
set -uo pipefail  # not -e: exit codes are managed explicitly below

payload="$(cat)"

# Read the edited path out of the tool payload. If jq is missing, fail open.
command -v jq >/dev/null 2>&1 || exit 0
file="$(printf '%s' "$payload" | jq -r '.tool_input.file_path // ""')"

[[ -n "$file" && "$file" == *.py && -f "$file" ]] || exit 0

project="${CLAUDE_PROJECT_DIR:-$PWD}"
marimo="$project/.venv/bin/marimo"
[[ -x "$marimo" ]] || exit 0

# Cheap gate before paying ~1s of interpreter startup. Most edited .py files are
# ordinary modules, and `marimo check` leaves those alone anyway — this just
# skips the wait. Same test `./go open` uses to decide what's a notebook.
grep -q 'marimo\.App(' "$file" || exit 0

before="$(cksum < "$file")"
"$marimo" check --fix --quiet "$file" >/dev/null 2>&1 || exit 0
after="$(cksum < "$file")"
[[ "$after" == "$before" ]] && exit 0

# The file on disk no longer matches what the tool just wrote, so say so: an edit
# built on the stale text would fail to apply, and the rewrite is worth knowing
# about rather than discovering as a mystery diff.
note="marimo check --fix rewrote $file after your edit; see /style-py."
jq -nc --arg note "$note" \
    '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $note}}'

exit 0

#!/usr/bin/env bash
#
# SubagentStart hook (settings.json-level, fires for every subagent) for local
# (devcontainer/non-web) sessions. Subagents get a fresh context and don't see
# the parent session's SessionStart output, so a subagent launching a
# memory-heavy diagnostic (e.g. loading a checkpoint) has no visibility into
# how little headroom this box has unless we tell it here.
#
# Unlike SessionStart, plain stdout on SubagentStart is NOT auto-injected as
# context — it must be JSON with hookSpecificOutput.additionalContext.
# https://code.claude.com/docs/en/hooks.md
#
set -euo pipefail

if [[ "${CLAUDE_CODE_REMOTE:-}" == 'true' ]]; then
    exit 0
fi

command -v free >/dev/null 2>&1 || exit 0

mem="$(free -h | awk '/^Mem:/ {print $7" free / "$2" total"}')"
swap="$(free -h | awk '/^Swap:/ {print $3" used / "$2" total"}')"
disk="$(df -h / 2>/dev/null | awk 'NR==2 {print $4" free / "$2" total ("$5" used)"}')"

note="Environment: RAM $mem · Swap $swap · Disk (/) $disk. Consider the available resources before launching jobs on this machine."

jq -n --arg note "$note" '{hookSpecificOutput: {hookEventName: "SubagentStart", additionalContext: $note}}'

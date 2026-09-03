#!/usr/bin/env bash
#
# SessionStart hook for local (devcontainer/non-web) Claude Code sessions.
#
# The web hook (session-start.sh) bootstraps tooling for the cloud runtime and
# is a no-op locally. This is its local counterpart: no tooling to bootstrap
# (the devcontainer handles that), just a small orientation note about the
# machine's resources. Model-default context reports platform/shell/OS but
# not RAM/swap/disk, and subagents doing memory-heavy work (e.g. loading a
# model checkpoint) have caused swap exhaustion before — this makes the
# headroom visible up front instead of discovered mid-task.
#
set -euo pipefail

# Local-only. The web hook covers CLAUDE_CODE_REMOTE=true.
if [[ "${CLAUDE_CODE_REMOTE:-}" == 'true' ]]; then
    exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# The two CLIs, generated rather than pinned here so they can't drift. Both are
# cheap: `./go` prints its usage in pure bash, and mini is called as the venv
# binary rather than through `uv run`, so nothing can trigger a sync. COLUMNS
# keeps argparse from wrapping the subcommand list; MINI_PROG makes the name it
# prints copy-pasteable. A bare invocation of either is a usage error, so they
# exit non-zero — `set +e` inside and `|| true` outside keep that from ending
# the hook here.
(
    set +e
    echo "## Project tooling"
    echo
    ./go 2>&1
    [[ -x .venv/bin/mini ]] && COLUMNS=200 MINI_PROG=bin/mini .venv/bin/mini 2>&1
) || true
echo

# Only the resource note below needs `free`; the tooling above lands either way.
command -v free >/dev/null 2>&1 || exit 0

mem="$(free -h | awk '/^Mem:/ {print $7" free / "$2" total"}')"
swap="$(free -h | awk '/^Swap:/ {print $3" used / "$2" total"}')"
disk="$(df -h / 2>/dev/null | awk 'NR==2 {print $4" free / "$2" total ("$5" used)"}')"

echo "## Environment"
echo
echo "RAM: $mem · Swap: $swap · Disk (/): $disk"
echo "Consider the available resources before launching jobs on this machine."

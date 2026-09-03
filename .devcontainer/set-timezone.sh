#!/usr/bin/env bash
# Adopt the host's time zone, recorded by host-timezone.sh just before the container
# started. Runs on every start (postStartCommand), so a host that has changed zones
# is picked up by a window reload — no rebuild needed.
#
# Only /etc/localtime is set, not a TZ environment variable: glibc and Python read
# /etc/localtime whenever TZ is unset, so this reaches every process in the container,
# including the ones VS Code spawns without going through a shell.

set -euo pipefail

tz_file="$(dirname "$0")/.host-tz"
tz=""
if [[ -s "$tz_file" ]]; then
    tz=$(tr -d '[:space:]' < "$tz_file")
fi
current=$(cat /etc/timezone 2>/dev/null || echo UTC)

if [[ -z "$tz" ]]; then
    echo "No host time zone recorded; staying on $current."
elif [[ ! -f "/usr/share/zoneinfo/$tz" ]]; then
    echo "Host time zone '$tz' is not in the container's zoneinfo database; staying on $current." >&2
elif [[ "$tz" != "$current" ]]; then
    sudo ln -snf "/usr/share/zoneinfo/$tz" /etc/localtime
    echo "$tz" | sudo tee /etc/timezone > /dev/null
    echo "Time zone set to $tz; it is now $(date)."
fi

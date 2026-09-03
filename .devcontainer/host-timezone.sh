#!/usr/bin/env sh
# Runs on the HOST via initializeCommand, before the container starts. Records the
# host's IANA zone name (e.g. "Australia/Sydney") where set-timezone.sh can read it
# from inside the container. POSIX sh: the host's shell isn't ours to choose.
#
# Must never exit non-zero — a failing initializeCommand stops the container from
# starting, and a missing time zone isn't worth that.

out="$(dirname "$0")/.host-tz"

# macOS: /etc/localtime -> /var/db/timezone/zoneinfo/Australia/Sydney
# Linux: /etc/localtime -> /usr/share/zoneinfo/Australia/Sydney
tz=$(readlink /etc/localtime 2>/dev/null | sed -e 's#.*/zoneinfo/##')
# Some hosts copy the zone file in rather than symlinking it.
[ -n "$tz" ] || tz=$(cat /etc/timezone 2>/dev/null)

printf '%s\n' "$tz" > "$out" 2>/dev/null
exit 0

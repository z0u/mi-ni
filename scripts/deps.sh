#!/usr/bin/env bash
#
# Dependency check: security advisories, GitHub Action pin freshness, and
# upgrades available for the packages we declare. Written for the weekly
# deps-routine agent (.claude/agents/deps-routine.md), and runnable by hand
# via `./go deps`.
#
# Read-only: nothing here writes uv.lock (the upgrade check is a --dry-run).
#
# Deliberately not `set -e`. Each section stands alone, so a section that can't
# run reports why and the others still produce their answer. If a section's
# output stops making sense, the tool underneath it has moved — fix the filter
# rather than working around it.

set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
cd "$PROJECT_ROOT"

# The `uv audit` JSON schema version these filters were written against. uv
# reports its own in `.schema.version`; a mismatch means the fields below may
# have moved, so the run says so instead of quietly reporting nothing.
AUDIT_SCHEMA='preview'

# OSV carries a GHSA *and* a PYSEC record for the same issue, each listing the
# shared CVE in `aliases`, so the sorted first element of id+aliases is a stable
# key for one underlying issue. Without this the counts read about 2x high.
# Where a group has both records, keep the GHSA one: several PYSEC entries come
# through with no summary.
read -r -d '' JQ_DEDUPE <<'JQ'
[ (.vulnerabilities // [])[] | . + {key: ([.id] + (.aliases // []) | sort | first)} ]
| group_by(.dependency.name + " " + .key)
| map((map(select(.id | startswith("GHSA"))) | first) // .[0])
JQ

usage() {
    echo "usage: ./go deps [--audit] [--actions] [--updates]"
    echo "       (no flags: all three)"
}

# --- Python advisories -------------------------------------------------------

check_python_advisories() {
    echo "## Python advisories (uv audit)"
    local json rc
    json=$(uv audit --frozen --output-format json 2>/dev/null)
    rc=$?
    # uv audit exits 1 when it finds something, which is a result rather than a
    # failure. Anything above that is the command itself not working.
    if (( rc > 1 )); then
        echo "  ⚠️  uv audit exited $rc. Re-run as \`uv audit --frozen\` to see the error."
        echo "      A tunnel error means api.osv.dev is not reachable from this environment."
        return 1
    fi

    local schema
    schema=$(jq -r '.schema.version // "missing"' <<<"$json")
    [[ "$schema" == "$AUDIT_SCHEMA" ]] \
        || echo "  ⚠️  schema is '$schema', these filters were written for '$AUDIT_SCHEMA' — check the fields still line up."

    local raw distinct audited adverse
    raw=$(jq -r '.summary.vulnerabilities' <<<"$json")
    audited=$(jq -r '.summary.audited_packages' <<<"$json")
    adverse=$(jq -r '.summary.adverse_statuses' <<<"$json")
    distinct=$(jq -r "$JQ_DEDUPE | length" <<<"$json" 2>/dev/null)
    [[ "$distinct" =~ ^[0-9]+$ ]] || distinct=0
    echo "  $audited packages audited; $raw records -> $distinct distinct issues; $adverse adverse statuses"

    # `adverse` covers PyPI project status: deprecation, and quarantine (which is
    # how a package pulled for malware shows up). Worth more attention than a CVE
    # in a library we only ever call locally.
    if (( adverse > 0 )); then
        echo
        echo "  Adverse project statuses:"
        jq -r '.adverse_statuses[] | "    \(.dependency.name) \(.dependency.version): \(.status // .kind // "?")"' <<<"$json"
    fi

    if (( distinct > 0 )); then
        echo
        jq -r "$JQ_DEDUPE"'
            | group_by(.dependency.name)
            | map({
                package: .[0].dependency.name,
                have:    .[0].dependency.version,
                issues:  length,
                fix:     ([.[].fix_versions[]] | unique
                           | max_by(split(".") | map(tonumber? // 0)))
              })
            | sort_by(-.issues)[]
            | "    \(.package) \(.have) -> \(.fix // "NO FIX")  (\(.issues) issue(s))"
        ' <<<"$json"
        echo
        echo "  Detail:"
        jq -r "$JQ_DEDUPE"'
            | sort_by(.dependency.name)[]
            | "    \(.dependency.name) \(.dependency.version)  \(.display_id)\n"
            + "      \(.summary // "(no summary)")\n"
            + "      fix: \(.fix_versions | join(", ") // "none")   \(.link)"
        ' <<<"$json"
    fi

    if (( raw > 0 && distinct == 0 )); then
        echo "  ⚠️  uv reported $raw records but the filter matched none — the output shape has changed."
        return 1
    fi
}

# --- Node advisories ---------------------------------------------------------

check_node_advisories() {
    echo "## Node advisories (npm audit)"
    local json
    # npm audit exits non-zero on findings too, so the exit code is not checked;
    # an unparseable body is the real failure.
    json=$(npm audit --json 2>/dev/null)
    if ! jq -e . >/dev/null 2>&1 <<<"$json"; then
        echo "  ⚠️  npm audit produced no parseable JSON. Re-run as \`npm audit\` to see why."
        return 1
    fi
    # `vulnerabilities` is an object keyed by package, and an empty one when
    # clean — hence the `[]?` guards rather than assuming an array.
    jq -r '
        "  \(.metadata.dependencies.total) packages; \(.metadata.vulnerabilities.total) vulnerabilities"
        , ( .vulnerabilities[]?
            | "    \(.name) [\(.severity)] fixAvailable=\(.fixAvailable)\n"
            + "      " + ([.via[]? | if type == "object" then .title else . end] | join("; ")) )
    ' <<<"$json"
}

# --- GitHub Action pins ------------------------------------------------------

check_actions() {
    echo "## GitHub Action pins"
    # `git ls-remote` reaches GitHub through the git proxy, so this needs no API
    # token — and unlike the GitHub MCP tools it is not scoped to this repo.
    # `sort -V` matters: plain ls-remote order is lexical, which puts v10 before v2.
    grep -rhoE 'uses: [A-Za-z0-9._-]+/[A-Za-z0-9._-]+@[^ ]+' .github/workflows/ \
        | sed 's/^uses: //' | sort -u \
        | while IFS='@' read -r repo pin; do
              local latest state
              latest=$(git ls-remote --tags --refs "https://github.com/$repo" 2>/dev/null \
                  | sed 's#.*refs/tags/##' \
                  | grep -E '^v?[0-9]+(\.[0-9]+)*$' \
                  | sort -V | tail -1)
              if   [[ -z "$latest"        ]]; then state='could not reach upstream'
              elif [[ "$pin" == "$latest" ]]; then state='current'
              elif [[ "$latest" == "$pin".* ]]; then state='current (floating tag)'
              else state="STALE -> $latest"
              fi
              printf '    %-28s %-10s %s\n' "$repo" "$pin" "$state"
          done
}

# --- Upgrades to packages we declare ----------------------------------------

check_declared_updates() {
    echo "## Upgrades available to declared dependencies"
    # `uv lock --upgrade --dry-run` lists every transitive bump (60+); these are
    # the ones we chose and would feel. It honours `exclude-newer`, so anything
    # published inside the cooldown window is absent by design.
    # Plain python3 rather than the venv: tomllib is stdlib from 3.11, so this
    # works before `uv sync` has run.
    local declared
    declared=$(python3 - <<'PY'
import re, tomllib

with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
specs = list(data['project'].get('dependencies', []))
for group in data.get('dependency-groups', {}).values():
    specs += [s for s in group if isinstance(s, str)]
names = {
    re.split(r'[\[<>=!~;\s]', spec, maxsplit=1)[0].strip().lower().replace('_', '-')
    for spec in specs
}
print('\n'.join(sorted(n for n in names if n)))
PY
)
    if [[ -z "$declared" ]]; then
        echo "  ⚠️  could not read dependency names out of pyproject.toml."
        return 1
    fi

    # uv writes these lines to stderr, hence the redirect.
    local out
    out=$(uv lock --upgrade --dry-run 2>&1 \
        | grep '^Update ' \
        | awk -v list="$declared" '
              BEGIN { n = split(list, a, "\n"); for (i = 1; i <= n; i++) want[a[i]] = 1 }
              { key = tolower($2); gsub(/_/, "-", key); if (key in want) print "    " $0 }
          ')
    if [[ -n "$out" ]]; then
        echo "$out"
    else
        echo "    (none)"
    fi
}

# --- Dispatch ----------------------------------------------------------------

want_audit=0 want_actions=0 want_updates=0
if (( $# == 0 )); then
    want_audit=1 want_actions=1 want_updates=1
fi
while (( $# )); do
    case "$1" in
        --audit)   want_audit=1 ;;
        --actions) want_actions=1 ;;
        --updates) want_updates=1 ;;
        -h|--help) usage; exit 0 ;;
        *)         echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

status=0
if (( want_audit )); then
    check_python_advisories || status=1
    echo
    check_node_advisories || status=1
    echo
fi
if (( want_actions )); then
    check_actions || status=1
    echo
fi
if (( want_updates )); then
    check_declared_updates || status=1
    echo
fi

if (( status != 0 )); then
    echo "⚠️  One or more checks could not run. See the notes above." >&2
fi
exit "$status"

#!/usr/bin/env bash

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
PROJECT_ROOT="$( cd -- "$( dirname -- "$SELF" )" &> /dev/null && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

# uv older than this can't parse the relative `exclude-newer` cooldown in
# pyproject.toml: it silently drops the cutoff, re-resolves, and rewrites
# uv.lock. Fail loudly instead. (`required-version` in [tool.uv] can't do this
# job, because the same parse failure discards that whole table.)
MIN_UV='0.11'
if command -v uv >/dev/null 2>&1; then
    have_uv="$(uv --version 2>/dev/null | awk '{print $2}')"
    if [[ "$(printf '%s\n%s\n' "$MIN_UV" "$have_uv" | sort -V | head -n1)" != "$MIN_UV" ]]; then
        echo "$SELF: uv $have_uv is too old (need >= $MIN_UV); it would re-resolve uv.lock. Upgrade: uv tool install uv --force (from PyPI; uv self update needs the GitHub API, which the web sandbox blocks)" >&2
        exit 1
    fi
fi

show_usage() {
    echo "usage: $SELF [-h] {install,auth,check,deps,open,lit,render,preview,publish,site,todo,worktrees} ..."
}

show_help() {
    show_usage
    # Important: heredoc indented with tab characters.
    cat <<-EOF
		New checkout? Start with: $0 install

		  install [--no-locked]:
		                       install dependencies (uv sync) and git hooks
		                       fails on a lockfile its manifest has outgrown, rather than
		                       re-resolving it; --no-locked lets it re-resolve
		  auth    [--check]:   set up credentials; --check just probes
		  check   [--lint] [--format] [--typecheck] [--test] [--links] [--fix]:
		                       run checks in parallel (default: all without --fix)
		                       individual commands: format | lint | types | tests | links
		                       advisory, and outside check: dead
		  links   [...paths]:  relative doc links and #anchors that no longer resolve
		                       (default: every .md we author)
		  deps    [--audit] [--actions] [--updates]:
		                       dependency review (default: all three) — advisories from
		                       uv audit and npm audit, Action pins against their newest
		                       upstream tag, and upgrades available to packages we declare.
		                       Read-only; the upgrade check is a --dry-run
		  open    <file>:      open a file in \$VISUAL/\$EDITOR (a report is a plain .py; pair
		                       it with \`lit serve\` for a live preview while you edit)
		  lit     render <doc.py> [--pdf] | serve <doc.py> [--port N]:
		                       weave a literate script (mini.lit: a .py with string prose
		                       between cells) to .mini/lit/<key>/, or serve it with live
		                       reload while you edit; preview/publish/render take it as a report
		  render  [...nbs] [--force]:
		                       render each report to readable Markdown at .mini/renders/<key>.md,
		                       figures linked beside it — for reading a report as a document
		                       (skips reports newer than their inputs; --force re-renders)
		  preview [...nbs] [--no-serve] [--force] [--port N]:
		                       export stale reports (each with a report.pdf beside its
		                       index.html, for review on paper or e-ink), assemble the site
		                       with local assets (never touches the network), and serve it
		  publish <nbs|--all>: export reports and sync their bundles (PDF included) to the
		                       publish tier
		  site:                assemble the public site from *published* bundles into _site/
		                       (for CI; read-only, never runs a report)
		  strays  [...paths]:  variable docstrings in a report, which weave as prose
		                       (default: docs/; also runs inside lint)
		  todo    [...sets] [--tag T] [--status S] [--bundle B] [--priority] [--grep RE] [--full]
		          [--tags] [--json] [--check]:
		                       list or search backlog items from todo/[set/]
		  worktrees [--prune] [--dry-run]:
		                       list agent worktrees; --prune removes the clean, landed ones

		Experiments are run with \`bin/mini\`, not \`$0\` — see \`bin/mini --help\`.
		EOF
}

case "${1:-}" in
    i|install)
        shift
        "$SCRIPT_DIR/install.sh" "$@"
        ;;
    auth)
        shift
        "$SCRIPT_DIR/auth.sh" "$@"
        ;;
    lit)
        shift
        uv run python -m mini.lit "$@"
        ;;
    format|formatting)
        shift
        "$SCRIPT_DIR/format.sh" "$@"
        ;;
    lint|linting|linters)
        shift
        "$SCRIPT_DIR/lint.sh" "$@"
        ;;
    dead|deadcode)
        shift
        "$SCRIPT_DIR/deadcode.sh" "$@"
        ;;
    deps|dependencies)
        shift
        "$SCRIPT_DIR/deps.sh" "$@"
        ;;
    link|links)
        shift
        uv run "$SCRIPT_DIR/check_md_links.py" "$@"
        ;;
    stray|strays)
        # A docstring hung under an assignment in a literate script weaves as a prose
        # paragraph. Part of `lint`, and separately runnable on one report.
        shift
        uv run "$SCRIPT_DIR/trailing_cell_docstrings.py" "$@"
        ;;
    type|types|typecheck)
        shift
        "$SCRIPT_DIR/typecheck.sh" "$@"
        ;;
    test|tests)
        shift
        "$SCRIPT_DIR/test.sh" "$@"
        ;;
    c|check)
        if [[ $# -gt 1 ]]; then
            shift
            "$SCRIPT_DIR/check.sh" "$@"
        else
            "$SCRIPT_DIR/check.sh" --lint --format --typecheck --test --links
        fi
        ;;
    o|edit|open)
        shift
        if [[ $# -eq 0 ]]; then
            echo "open what? pass a file (opens in \$VISUAL/\$EDITOR)." 1>&2
            exit 2
        fi
        editor="${VISUAL:-${EDITOR:-code}}"
        if ! command -v "$editor" > /dev/null; then
            echo "no editor: '$editor' not found — set \$VISUAL or \$EDITOR" 1>&2
            exit 127
        fi
        ( set -x; "$editor" "$@" )
        ;;
    render)
        # One report at a time: rendering runs the report's cells, so a bare `render` over
        # every report would be compute nobody asked for. Flags pass through to the script
        # (--force, --assets-dir DIR).
        shift
        nbs=() flags=()
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --assets-dir) flags+=("$1" "${2:?--assets-dir needs a value}"); shift ;;
                -*) flags+=("$1") ;;
                *) nbs+=("$1") ;;
            esac
            shift
        done
        if [[ ${#nbs[@]} -eq 0 ]]; then
            echo "render what? name one or more reports, e.g." 1>&2
            echo "  $0 render docs/pipeline/report.py" 1>&2
            exit 2
        fi
        for nb in "${nbs[@]}"; do
            ( set -x; uv run "$SCRIPT_DIR/export_report_md.py" "$nb" "${flags[@]}" )
        done
        ;;
    p|preview)
        shift
        serve=1 port=8000 stale=--stale-only
        nbs=()
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --no-serve) serve=0 ;;
                --force) stale= ;;
                --port) port="${2:?--port needs a value}"; shift ;;
                -*) echo "preview: unknown flag '$1' (flags: --no-serve --force --port N)" 1>&2; exit 2 ;;
                *) nbs+=("$1") ;;
            esac
            shift
        done
        ( set -x; uv run "$SCRIPT_DIR/export_reports.py" ${stale:+"$stale"} "${nbs[@]}" )
        ( set -x; uv run "$SCRIPT_DIR/build_site.py" --localize )
        if [[ $serve -eq 1 ]]; then
            ( set -x; uv run "$SCRIPT_DIR/preview_server.py" "$PROJECT_ROOT/_site" "$port" )
        else
            echo
            echo "Site assembled at _site/ (bundles in .mini/exports/)."
            echo "Serve it later with: $0 preview  — or render a bundle headlessly (report-render skill)."
        fi
        ;;
    publish)
        shift
        # Export each named report and mirror its bundle to the publish tier (needs ./go auth).
        # Explicit by design: export_reports.py refuses a bare --publish without names or --all.
        ( set -x; uv run "$SCRIPT_DIR/export_reports.py" --publish "$@" )
        ;;
    site)
        shift
        uv run "$SCRIPT_DIR/build_site.py" --externalize "$@"
        ;;
    todo)
        shift
        uv run "$SCRIPT_DIR/todo.py" "$@"
        ;;
    worktrees|worktree|wt)
        shift
        uv run "$SCRIPT_DIR/worktrees.py" "$@"
        ;;
    e|export|r|run|s|serve|build|scrub|clean)
        case "$1" in
            e|export)     echo "'export' is gone — '$0 preview --no-serve' exports stale reports to .mini/exports/" ;;
            r|run)        echo "'run' is gone — 'bin/mini run <experiment.py>' runs experiments; '$0 preview' renders reports; 'uv run ...' for anything else" ;;
            s|serve)      echo "'serve' is gone — '$0 preview' exports what's stale, then builds and serves" ;;
            build)        echo "'build' split in two — '$0 preview' assembles locally; '$0 site' assembles the public site from published bundles (CI)" ;;
            scrub|clean)  echo "'scrub' is gone — a literate script's export has nothing to scrub" ;;
        esac 1>&2
        exit 2
        ;;
    h|help|-h|--help)
        show_help
        exit 0
        ;;
    *)
        show_usage 1>&2
        exit 2
        ;;
esac

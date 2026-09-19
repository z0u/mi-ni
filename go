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
    echo "usage: $SELF [-h] {install,auth,check,deps,render,serve,preview,publish,site,todo,worktrees} ..."
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
		  render  [...reports] [--pdf]:
		                       weave each report (mini.lit: a .py with string prose between
		                       cells) to .mini/lit/<key>/ — index.html and index.md, figures
		                       beside them under _assets/ — for reading it as a document
		  serve   <report> [--port N]:
		                       serve one report with live reload while you edit it
		  preview [...reports] [--no-serve] [--force] [--port N]:
		                       export stale reports, assemble the site with local assets
		                       (never touches the network; each report printed to
		                       _site/<key>/report.pdf for review on paper or e-ink,
		                       unchanged ones reused from .mini/pdfs/), and serve it
		  publish <reports|--all>:
		                       export reports and sync their bundles to the publish tier
		  site:                assemble the public site from *published* bundles into _site/
		                       (for CI; read-only, never runs a report; prints each PDF,
		                       reusing the previous deploy's via MINI_PDF_MEMO)
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
    render)
        # Named reports only: rendering runs the report's cells, so a bare `render` over
        # every report would be compute nobody asked for. Flags pass through (--pdf).
        shift
        paths=() flags=()
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -*) flags+=("$1") ;;
                *) paths+=("$1") ;;
            esac
            shift
        done
        if [[ ${#paths[@]} -eq 0 ]]; then
            echo "render what? name one or more reports, e.g." 1>&2
            echo "  $0 render docs/pipeline/report.py" 1>&2
            exit 2
        fi
        for path in "${paths[@]}"; do
            ( set -x; uv run python -m mini.lit render "$path" "${flags[@]}" )
        done
        ;;
    serve)
        shift
        ( set -x; uv run python -m mini.lit serve "$@" )
        ;;
    p|preview)
        shift
        serve=1 port=8000 stale=--stale-only
        paths=()
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --no-serve) serve=0 ;;
                --force) stale= ;;
                --port) port="${2:?--port needs a value}"; shift ;;
                -*) echo "preview: unknown flag '$1' (flags: --no-serve --force --port N)" 1>&2; exit 2 ;;
                *) paths+=("$1") ;;
            esac
            shift
        done
        ( set -x; uv run "$SCRIPT_DIR/export_reports.py" ${stale:+"$stale"} "${paths[@]}" )
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
    e|export|r|run|build|scrub|clean|lit|o|edit|open)
        case "$1" in
            e|export)     echo "'export' is gone — '$0 preview --no-serve' exports stale reports to .mini/exports/" ;;
            r|run)        echo "'run' is gone — 'bin/mini run <experiment.py>' runs experiments; '$0 preview' renders reports; 'uv run ...' for anything else" ;;
            build)        echo "'build' split in two — '$0 preview' assembles locally; '$0 site' assembles the public site from published bundles (CI)" ;;
            scrub|clean)  echo "'scrub' is gone — a literate script's export has nothing to scrub" ;;
            lit)          echo "'lit' is gone — '$0 render <report>' weaves a literate script; '$0 serve <report>' serves it with live reload" ;;
            o|edit|open)  echo "'open' is gone — a report is a plain .py, so open it in your editor; '$0 serve <report>' previews it live" ;;
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

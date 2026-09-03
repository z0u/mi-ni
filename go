#!/usr/bin/env bash

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
PROJECT_ROOT="$( cd -- "$( dirname -- "$SELF" )" &> /dev/null && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

is_marimo_notebook() {
    [[ "${1:-}" == *.py && -f "${1:-}" ]] && grep -q 'marimo\.App(' "$1"
}

show_usage() {
    echo "usage: $SELF [-h] {install,auth,check,open,render,preview,publish,site,worktrees} ..."
}

show_help() {
    show_usage
    # Important: heredoc indented with tab characters.
    cat <<-EOF
		New checkout? Start with: $0 install

		  install [--locked]:  install dependencies (uv sync) and git hooks
		                       --locked fails on a lockfile its manifest has outgrown,
		                       rather than re-resolving it (the default under \$CI)
		  auth    [--check]:   set up credentials; --check just probes
		  check   [--lint] [--format] [--typecheck] [--test] [--links] [--fix]:
		                       run checks in parallel (default: all without --fix)
		                       individual commands: format | lint | types | tests | links
		                       advisory, and outside check: dead | annotations
		  links   [...paths]:  relative doc links and #anchors that no longer resolve
		                       (default: every .md we author)
		  open    <file> [--browser]:
		                       open a Marimo notebook for live editing — watches the file so
		                       the IDE stays the editor, and prints a URL that lands in the
		                       app view; anything else opens in \$EDITOR
		  render  [...nbs] [--force]:
		                       render each report to readable Markdown at .mini/renders/<key>.md,
		                       figures as ![alt](path) links — for reading a report as a document
		                       (skips reports newer than their inputs; --force re-renders)
		  preview [...nbs] [--no-serve] [--force] [--port N]:
		                       export stale reports, assemble the site with local assets
		                       (never touches the network), and serve it
		  publish <nbs|--all>: export reports and sync their bundles to the publish tier
		  site:                assemble the public site from *published* bundles into _site/
		                       (for CI; read-only, never runs a notebook)
		  strays  [...paths]:  Marimo cells that end on a docstring, which publishes it as
		                       the cell's output (default: docs/; also runs inside lint)
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
    link|links)
        shift
        uv run "$SCRIPT_DIR/check_md_links.py" "$@"
        ;;
    stray|strays)
        # Cells whose last statement is a docstring, which Marimo publishes as the
        # cell's output. Part of `lint`, and separately runnable on one notebook.
        shift
        uv run "$SCRIPT_DIR/trailing_cell_docstrings.py" "$@"
        ;;
    ann|annotations)
        # Advisory, like `dead`: a worklist of public cell variables that reach
        # downstream cells untyped. Deliberately outside `check` while the backlog
        # is longer than a branch should carry.
        shift
        uv run "$SCRIPT_DIR/unannotated_cell_vars.py" "$@"
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
            echo "open what? pass a Marimo notebook (opens in marimo edit --watch)," 1>&2
            echo "or any other file (opens in \$VISUAL/\$EDITOR)." 1>&2
            exit 2
        fi
        # Any position, so `--browser` and marimo's own flags can precede the file.
        notebook=
        for arg in "$@"; do
            if is_marimo_notebook "$arg"; then notebook="$arg"; break; fi
        done
        if [[ -n "$notebook" ]]; then
            ( set -x; uv run "$SCRIPT_DIR/edit_notebook.py" "$@" )
        else
            editor="${VISUAL:-${EDITOR:-code}}"
            if ! command -v "$editor" > /dev/null; then
                echo "no editor: '$editor' not found — set \$VISUAL or \$EDITOR" 1>&2
                exit 127
            fi
            ( set -x; "$editor" "$@" )
        fi
        ;;
    render)
        # One notebook at a time: rendering re-runs the notebook, so a bare `render` over
        # every report would be an hour's compute nobody asked for. Flags pass through to
        # the script (--force, --no-reflow, --sandbox, --timeout N).
        shift
        nbs=() flags=()
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --timeout) flags+=("$1" "${2:?--timeout needs a value}"); shift ;;
                -*) flags+=("$1") ;;
                *) nbs+=("$1") ;;
            esac
            shift
        done
        if [[ ${#nbs[@]} -eq 0 ]]; then
            echo "render what? name one or more report notebooks, e.g." 1>&2
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
            scrub|clean)  echo "'scrub' is internal now (export applies it) — scripts/clean_docs.py if you really need it" ;;
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

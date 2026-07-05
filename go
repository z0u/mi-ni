#!/usr/bin/env bash

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
PROJECT_ROOT="$( cd -- "$( dirname -- "$SELF" )" &> /dev/null && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

is_marimo_notebook() {
    [[ "${1:-}" == *.py && -f "${1:-}" ]] && grep -q 'marimo\.App(' "$1"
}

case "${1:-all}" in
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
            "$SCRIPT_DIR/check.sh" --lint --format --typecheck --test
        fi
        ;;
    e|export|r|run)
        # `run` is a deprecated alias: it collides with `mini run` (which executes
        # compute) whereas this only exports a static bundle. Prefer `export`.
        shift
        if is_marimo_notebook "${1:-}"; then
            # Export the report to its bundle (.mini/exports/<key>/); preview via ./go build.
            ( set -x; uv run "$SCRIPT_DIR/export_reports.py" "$1" )
        else
            ( set -x; uv run "$@" )
        fi
        ;;
    publish)
        shift
        # Export each report and mirror its bundle to the HF bucket (needs ./go auth).
        ( set -x; uv run "$SCRIPT_DIR/export_reports.py" --publish "$@" )
        ;;
    o|edit|open)
        shift
        if is_marimo_notebook "${1:-}"; then
            uv run marimo edit "$@"
        else
            "${VISUAL:-${EDITOR:-code}}" "$@"
        fi
        ;;
    build|site)
        shift
        uv run "$SCRIPT_DIR/build_site.py" "$@"
        ;;
    scrub|clean)
        # `clean` is a deprecated alias: in most build tools it deletes outputs, but
        # this scrubs terminal control sequences and redacts Modal URLs. Prefer `scrub`.
        shift
        "$SCRIPT_DIR/clean_docs.py" "$@"
        ;;
    s|serve)
        "$SELF" build
        npx serve -n "$PROJECT_ROOT/_site"
        ;;
    *)
        # Important: heredoc indented with tab characters.
        cat <<-EOF 1>&2
			Usage: $0 {auth|check|lint|format|types|tests|export|open|publish|build|scrub|serve}
			  install:           install dependencies (uv sync) and git hooks
			  auth   [--check]:  set up credentials, or --check to just probe & print their status
			  check  [...args]:  run all checks in parallel (--lint --format --typecheck --test --fix)
			  format [...args]:  format code (ruff format)
			  lint   [...args]:  run linters (ruff check)
			  types  [...args]:  check types (ty)
			  tests  [...args]:  run tests (pytest)
			  export [...args]:  export a Marimo report to its bundle, or run anything else through uv
			  open   [...args]:  open a Marimo notebook in Marimo, or anything else in \$EDITOR
			  publish [...nbs]:  export reports and mirror their bundles to the HF bucket
			  build  [...args]:  build the static site (from synced bundles, or local for offline)
			  scrub  [...args]:  scrub terminal control sequences / redact Modal URLs from Marimo HTML
			  serve:             build and serve at http://localhost:8000
			  (aliases: \`run\`→\`export\`, \`clean\`→\`scrub\` — deprecated)
			EOF
        exit 1
        ;;
esac

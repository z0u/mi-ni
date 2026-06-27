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
    r|run)
        shift
        if is_marimo_notebook "${1:-}"; then
            notebook="$1"
            shift
            # Allow an explicit '--' before notebook args (./go run nb -- --app=modal);
            # we add our own separator below, so drop a redundant leading one.
            [[ "${1:-}" == "--" ]] && shift
            out="$( dirname -- "$notebook" )/__marimo__/$( basename -- "$notebook" .py ).html"
            ( set -x; uv run marimo export html -f "$notebook" -o "$out" -- "$@" )
        else
            ( set -x; uv run "$@" )
        fi
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
        "$SCRIPT_DIR/clean_docs.py"
        uv run "$SCRIPT_DIR/build_site.py" "$@"
        ;;
    clean)
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
			Usage: $0 {check|lint|format|types|tests|run|open|build|clean|serve}
			  install:           install dependencies (uv sync) and git hooks
			  check  [...args]:  run all checks in parallel (--lint --format --typecheck --test --fix)
			  format [...args]:  format code (ruff format)
			  lint   [...args]:  run linters (ruff check)
			  types  [...args]:  check types (ty)
			  tests  [...args]:  run tests (pytest)
			  run    [...args]:  run & export a Marimo notebook, or run anything else through uv
			  open   [...args]:  open a Marimo notebook in Marimo, or anything else in \$EDITOR
			  build  [...args]:  build static site
			  clean  [...args]:  clean Marimo HTML/session output (apply control chars)
			  serve:             clean docs and serve at http://localhost:8000
			EOF
        exit 1
        ;;
esac

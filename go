#!/usr/bin/env bash

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
PROJECT_ROOT="$( cd -- "$( dirname -- "$SELF" )" &> /dev/null && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

case "${1:-all}" in
    install)
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
    type|types)
        shift
        "$SCRIPT_DIR/typecheck.sh" "$@"
        ;;
    test|tests)
        shift
        "$SCRIPT_DIR/test.sh" "$@"
        ;;
    check)
        if [[ $# -gt 1 ]]; then
            shift
            "$SCRIPT_DIR/check.sh" "$@"
        else
            "$SCRIPT_DIR/check.sh" --lint --format --typecheck --test
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
    serve)
        "$SELF" build
        npx serve -n "$PROJECT_ROOT/_site"
        ;;
    *)
        # Important: heredoc indented with tab characters.
        cat <<-EOF 1>&2
			Usage: $0 {check|lint|format|types|tests|build|clean|serve}
			  install:           install dependencies (uv sync) and git hooks
			  check  [...args]:  run all checks in parallel (--lint --format --typecheck --test --fix)
			  format [...args]:  format code (ruff format)
			  lint   [...args]:  run linters (ruff check)
			  types  [...args]:  check types (ty)
			  tests  [...args]:  run tests (pytest)
			  build  [...args]:  build static site
			  clean  [...args]:  clean Marimo HTML/session output (apply control chars)
			  serve:             clean docs and serve at http://localhost:8000
			EOF
        exit 1
        ;;
esac

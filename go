#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/scripts"

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
        "$SCRIPT_DIR/format.sh"
        "$SCRIPT_DIR/lint.sh"
        "$SCRIPT_DIR/typecheck.sh"
        "$SCRIPT_DIR/test.sh"
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
        "$SCRIPT_DIR/clean_docs.py"
        python3 -m http.server 8000 -d "$(dirname "$SCRIPT_DIR")/docs/__marimo__"
        ;;
    *)
        # Important: heredoc indented with tab characters.
        cat <<-EOF 1>&2
			Usage: $0 {check|lint|format|types|tests|build|clean|serve}
			  install:           install dependencies (uv sync) and git hooks
			  check:             run all checks
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

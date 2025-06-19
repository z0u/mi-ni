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
        "$SCRIPT_DIR/deadcode.sh"
        ;;
    build|site)
        shift
        "$SCRIPT_DIR/build_site.py" "$@"
        ;;
    *)
        # Important: heredoc indented with tab characters.
        cat <<-EOF 1>&2
			Usage: $0 {check|lint|format|types|tests|build}
			  install:           install dependencies (uv sync)
			  check:             run all checks
			  format [...args]:  format code (ruff format)
			  lint   [...args]:  run linters (ruff check)
			  types  [...args]:  check types (pyright)
			  tests  [...args]:  run tests (pytest)
			  build  [...args]:  build static site
			EOF
        exit 1
        ;;
esac

#!/usr/bin/env bash

set -euo pipefail

lint() {
    (
        set -x
        uv run ruff check "$@"
    )
}

format() {
    (
        set -x
        uv run ruff format "$@"
    )
}

check_types() {
    (
        set -x
        uv run pyright "$@"
    )
}

run_tests() {
    (
        set -x
        uv run pytest "$@"
    )
}

case "${1:-all}" in
    format|formatting)
        shift
        format "$@"
        ;;
    lint|linting|linters)
        shift
        lint "$@"
        ;;
    type|types)
        shift
        check_types "$@"
        ;;
    test|tests)
        shift
        run_tests "$@"
        ;;
    all)
        format
        lint
        check_types
        run_tests
        ;;
    *)
        # Important: heredoc indented with tab characters.
        cat <<-EOF 1>&2
			Usage: $0 {all|lint|format|types|tests}
			  all:               run all checks
			  format [...args]:  format code (ruff format)
			  lint   [...args]:  run linters (ruff check)
			  types  [...args]:  check types (pyright)
			  tests  [...args]:  run tests (pytest)
			EOF
        exit 1
        ;;
esac

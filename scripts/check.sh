#!/usr/bin/env bash

set -euo pipefail

lint() {
    (
        set -x
        uv run ruff check "$@"
    )
}

deadcode() {
    (
        set -x
        mkdir -p .vulture-cache
        rm -r .vulture-cache/* || true
        uv run python scripts/ipynb_to_py.py *.ipynb .vulture-cache/
        uv run vulture "$@"
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
    dead|deadcode)
        shift
        deadcode "$@"
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
        deadcode
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

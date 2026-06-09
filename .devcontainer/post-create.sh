#!/usr/bin/env bash

set -euo pipefail

(
    set -x

    # https://github.com/orgs/community/discussions/43534
    sudo cp .devcontainer/welcome.txt /usr/local/etc/vscode-dev-containers/first-run-notice.txt

    # Make the volume mounts writable.
    sudo chown -R "$USER:$USER" ~/
    sudo chown -R "$USER:$USER" .venv
    sudo chown -R "$USER:$USER" node_modules

    # Seed default configs if not already present
    [[ -f ~/.config/marimo/marimo.toml ]] || cp .devcontainer/marimo.toml ~/.config/marimo/marimo.toml

    # Initialize Python environment.
    uv venv --allow-existing < /dev/null
    ./go install < /dev/null

    npx skills add marimo-team/marimo-pair --yes < /dev/null
)

echo "Virtual environment created. You may need to restart the Python language server."

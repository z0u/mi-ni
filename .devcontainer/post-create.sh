#!/usr/bin/env bash

set -euo pipefail

(
    set -x

    # https://github.com/orgs/community/discussions/43534
    sudo cp .devcontainer/welcome.txt /usr/local/etc/vscode-dev-containers/first-run-notice.txt

    # Initialize git LFS hooks for this repository (see .gitattributes)
    git lfs install

    # Make the volume mounts writable. Even though the uv cache is a subdirectory, the parent is created by Docker as root, so we need to change the owner of that too.
    sudo chown -R "$USER:$USER" ~/.cache
    sudo chown -R "$USER:$USER" .venv
    sudo chown -R "$USER:$USER" ~/.config/marimo
    sudo chown -R "$USER:$USER" ~/.claude

    # Seed default configs if not already present
    [[ -f ~/.config/marimo/marimo.toml ]] || cp .devcontainer/marimo.toml ~/.config/marimo/marimo.toml

    # Initialize Python environment.
    uv venv --allow-existing < /dev/null
    ./go install < /dev/null

    # Install Claude Code here rather than in Dockerfile, because it updates itself frequently.
    curl -fsSL https://claude.ai/install.sh | bash

    # Agent skills — install to user home to avoid conflicts with project skills.
    npx skills add marimo-team/marimo-pair -g --agent claude-code --yes
)

echo "Virtual environment created. You may need to restart the Python language server."

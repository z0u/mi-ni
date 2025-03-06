#!/usr/bin/env bash

set -euo pipefail

# Make the volume mounts writable. Even though the uv cache is a subdirectory, the parent is created by Docker as root, so we need to change the owner of that too.
sudo chown -R "$USER:$USER" ~/.cache
sudo chown -R "$USER:$USER" .venv

# Initialize Python environment.
uv venv --allow-existing
uv sync --all-groups
echo "Virtual environment created. You may need to restart the Python language server."

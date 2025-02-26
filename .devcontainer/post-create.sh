#!/usr/bin/env bash

set -euo pipefail

# Make the volume mounts writable.
sudo chown -R "$USER:$USER" /home/vscode/.cache/uv
sudo chown -R "$USER:$USER" .venv

# Initialize Python environment.
uv venv --allow-existing
uv sync
echo "Virtual environment created. You may need to restart the Python language server." >&2
